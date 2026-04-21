import os
import sys
import numpy as np
from gymnasium import spaces
import torch
from omnisafe.envs.core import CMDP, ClassVar, env_register, env_unregister
import omnisafe

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from wrapper import ReportWrapper
from config_loader import load_scenario
from scenario_creator import create_env_from_config

# Default configuration - SINGLE SCENARIO only
DEFAULT_SCENARIO            = 'medium'  # Only medium scenario
DEFAULT_STEPS_PER_EPISODE   = 600       # Steps per episode
DEFAULT_TOTAL_STEPS         = 600 * 100  # 600 steps * 100 epochs = 60,000 total steps
DEFAULT_PENALTY             = 10.0
DEFAULT_RESULTS_PATH        = './pipelines/cpo/'
DEFAULT_STEPS_PER_EPOCH     = 600 * 60   # 60 episodes per epoch = 36,000 steps per epoch

ENV_ID = "RanSliceCPOSingleScenario-v0"

# Module-level mutable state
_rng                    = np.random.default_rng(3)
_penalty                = DEFAULT_PENALTY
_total_steps            = DEFAULT_TOTAL_STEPS
_steps_per_episode      = DEFAULT_STEPS_PER_EPISODE
_results_path           = DEFAULT_RESULTS_PATH
_scenario               = DEFAULT_SCENARIO  # Single scenario, not a list
_steps_per_epoch        = DEFAULT_STEPS_PER_EPOCH
_current_run_id         = 0
_env_instance_count     = {}
_scenario_configs_cache = {}
_epoch_counter          = 0


def _get_scenario_config(name):
    if name not in _scenario_configs_cache:
        _scenario_configs_cache[name] = load_scenario('scenarios.yaml', name)
    return _scenario_configs_cache[name]


@env_register
@env_unregister
class RanSliceCPOEnv(CMDP):
    """CPO environment using raw NodeB state - single scenario (medium only)."""

    _support_envs: ClassVar[list[str]] = [ENV_ID]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs               = 1

    def __init__(self, env_id: str, **kwargs) -> None:
        global _env_instance_count
        super().__init__(env_id)

        self._run_id = _current_run_id
        if self._run_id not in _env_instance_count:
            _env_instance_count[self._run_id] = 0
        _env_instance_count[self._run_id] += 1
        self._instance_id = _env_instance_count[self._run_id]
        self._is_training_env = (self._instance_id == 1)

        print(f"[Run {self._run_id}] Creating CPO env instance {self._instance_id} "
              f"({'TRAINING' if self._is_training_env else 'EVAL'})")
        print(f"Scenario: {_scenario} (no switching)")
        print(f"Steps per episode: {_steps_per_episode}")
        print(f"Steps per epoch: {_steps_per_epoch} (60 episodes)")
        print(f"Total steps: {_total_steps} (100 epochs)")

        # Load single scenario config
        self.scenario_config = _get_scenario_config(_scenario)
        self._global_step_count = 0

        # Create environment with medium scenario
        raw_env = create_env_from_config(self.scenario_config, _rng, penalty=_penalty)

        self._results_path = _results_path
        os.makedirs(self._results_path, exist_ok=True)

        # Wrap with ReportWrapper for logging
        if self._is_training_env:
            self._env = ReportWrapper(
                raw_env, steps=_total_steps, control_steps=5000,
                env_id=str(self._run_id), path=self._results_path,
                verbose=False, continuous_mode=True,
            )
        else:
            self._env = ReportWrapper(
                raw_env, steps=_total_steps, control_steps=_total_steps + 1,
                env_id=f"{self._run_id}_eval", path=self._results_path,
                verbose=False, continuous_mode=True,
            )

        self._max_episode_steps = _steps_per_episode
        self._n_prbs   = self.scenario_config.n_prbs
        self._n_slices = self._env.n_slices
        self._step_count = 0
        
        # Get the raw state dimension from NodeB
        self._state_dim = self._env.n_variables
        
        # Action space: PRB allocation per slice (integers)
        self._action_space = spaces.Box(low=0, high=self._n_prbs,
                                        shape=(self._n_slices,), dtype=np.int64)
        
        # Observation space: raw normalized state from NodeB
        self._observation_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                             shape=(self._state_dim,), dtype=np.float32)
        
        self._last_allocation = np.zeros(self._n_slices, dtype=int)

        # Emergency PRB redistribution state
        self._consecutive_violations = np.zeros(self._n_slices, dtype=int)
        self._priority_order   = [2, 0, 1]   # URLLC > eMBB > mMTC
        self._MMTC_IDX         = 1
        self._MMTC_MIN_PRBS    = 5

    # Reset
    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._consecutive_violations.fill(0)
        result = self._env.reset()
        # Handle different return formats
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        
        # obs is already the raw NodeB state (normalized metrics)
        return torch.as_tensor(obs, dtype=torch.float32), info

    # Step with proactive PRB redistribution
    def step(self, action: torch.Tensor):
        # Convert action to numpy and ensure integer PRB allocation
        act = action.cpu().numpy()
        act = np.clip(act, 0, self._n_prbs).astype(np.int64)
        
        # Ensure total PRBs don't exceed available
        if act.sum() > self._n_prbs:
            # Scale down proportionally
            act = (act / act.sum() * self._n_prbs).astype(np.int64)
            # Fix rounding errors
            while act.sum() > self._n_prbs:
                max_idx = np.argmax(act)
                act[max_idx] = max(0, act[max_idx] - 1)
            while act.sum() < self._n_prbs and self._n_prbs - act.sum() > 0:
                # Distribute remaining PRBs to highest priority slices
                for s in self._priority_order:
                    if act.sum() < self._n_prbs:
                        act[s] += 1
        
        # Proactive redistribution: check consecutive violations
        suffering = [s for s in self._priority_order 
                     if self._consecutive_violations[s] > 0 and s != self._MMTC_IDX]
        
        if suffering:
            target = suffering[0]  # highest-priority suffering slice
            # Steal from mMTC (keep minimum PRBs)
            stealable = max(0, act[self._MMTC_IDX] - self._MMTC_MIN_PRBS)
            if stealable > 0:
                act[self._MMTC_IDX] -= stealable
                act[target] += stealable
        
        # Take step in environment
        result = self._env.step(act)
        
        # Handle different return formats
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
        else:
            state, reward, terminated, truncated, info = result[0], result[1], False, False, {}
        
        # Extract cost from violations
        violations = info.get('violations', np.zeros(self._n_slices))
        if hasattr(violations, '__len__') and len(violations) == self._n_slices:
            cost = float(violations.sum())
            # Update consecutive violation counters
            for s in range(self._n_slices):
                if violations[s] > 0:
                    self._consecutive_violations[s] += 1
                else:
                    self._consecutive_violations[s] = 0
        else:
            cost = float(info.get('total_violations', 0.0))
        
        self._step_count += 1
        self._global_step_count += 1
        
        # Reseed RNG at epoch boundaries
        global _epoch_counter, _rng
        if self._is_training_env and self._global_step_count % _steps_per_epoch == 0:
            _epoch_counter += 1
            _rng = np.random.default_rng(3 + _epoch_counter * 1000 + self._run_id)
            print(f"\n=== EPOCH {_epoch_counter} COMPLETED ===")
        
        # Episode termination
        if self._step_count >= _steps_per_episode:
            truncated = True
            self._step_count = 0
        
        # Handle episode end
        if terminated or truncated:
            final_state = torch.as_tensor(state, dtype=torch.float32)
            result = self._env.reset()
            if isinstance(result, tuple):
                state, _ = result
            else:
                state = result
            info["final_observation"] = final_state
        else:
            info["final_observation"] = torch.as_tensor(state, dtype=torch.float32)
        
        return (
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            info,
        )

    # Required abstract methods
    def render(self, *args, **kwargs):
        """Render the environment."""
        if hasattr(self._env, "render"):
            return self._env.render()
        return None

    def set_seed(self, seed: int) -> None:
        """Set the random seed for the environment."""
        global _rng
        _rng = np.random.default_rng(seed)
        if hasattr(self._env, "set_seed"):
            self._env.set_seed(seed)

    def close(self) -> None:
        """Close the environment and save results."""
        if self._is_training_env and hasattr(self._env, 'save_results'):
            self._env.save_results()
        if hasattr(self._env, 'close'):
            self._env.close()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps


class TrainerCPO:
    """Train CPO agent on single scenario (medium only)."""

    def __init__(self, results_path=DEFAULT_RESULTS_PATH,
                 total_steps=None,
                 steps_per_episode=DEFAULT_STEPS_PER_EPISODE,
                 scenario=DEFAULT_SCENARIO,
                 penalty=DEFAULT_PENALTY,
                 cost_limit=100.0,
                 steps_per_epoch=None,
                 num_epochs=100,
                 device="cpu"):
        
        self.steps_per_episode = steps_per_episode
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch if steps_per_epoch else steps_per_episode * 60
        self.total_steps = total_steps if total_steps else self.steps_per_epoch * num_epochs
        
        self.results_path = results_path
        self.scenario = scenario
        self.penalty = penalty
        self.cost_limit = cost_limit
        self.device = device
        os.makedirs(self.results_path, exist_ok=True)

    def train(self, run_id: int):
        global _current_run_id, _env_instance_count, _total_steps
        global _steps_per_episode, _results_path, _scenario, _penalty
        global _steps_per_epoch, _epoch_counter
        
        _current_run_id      = run_id
        _env_instance_count[run_id] = 0
        _total_steps         = self.total_steps
        _steps_per_episode   = self.steps_per_episode
        _results_path        = self.results_path
        _scenario            = self.scenario
        _penalty             = self.penalty
        _steps_per_epoch     = self.steps_per_epoch
        _epoch_counter       = 0

        print(f'\n{"="*60}')
        print(f'=== CPO Training Run {run_id} ===')
        print(f'{"="*60}')
        print(f'Scenario: {self.scenario} (NO SWITCHING)')
        print(f'Steps per episode: {self.steps_per_episode}')
        print(f'Steps per epoch: {self.steps_per_epoch} (60 episodes)')
        print(f'Number of epochs: {self.num_epochs}')
        print(f'Total steps: {self.total_steps}')
        print(f'Cost limit: {self.cost_limit}')
        print(f'Device: {self.device}')
        print(f'Output: {self.results_path}history_{run_id}.npz')

        custom_cfgs = {
            "train_cfgs": {"total_steps": self.total_steps, "device": self.device},
            "algo_cfgs": {
                "steps_per_epoch": self.steps_per_epoch,
                "update_iters": 3,
                "batch_size": 128,
                "target_kl": 0.02,
                "entropy_coef": 0.01,
                "cost_limit": self.cost_limit,
                "use_max_grad_norm": True,
                "max_grad_norm": 0.5,
                "gamma": 0.99,
                "cost_gamma": 0.99,
                "lam": 0.95,
                "lam_c": 0.95,
                "cg_damping": 0.1,
                "cg_iters": 15,
                "use_cost": True,
                "obs_normalize": True,
                "reward_normalize": True,
                "cost_normalize": True,
            },
            "model_cfgs": {
                "actor": {"hidden_sizes": [256, 256], "activation": "tanh"},
                "critic": {"hidden_sizes": [256, 256], "activation": "tanh", "lr": 3e-4},
            },
            "logger_cfgs": {"use_wandb": False, "save_model_freq": 1},
        }

        agent = omnisafe.Agent(algo="CPO", env_id=ENV_ID, custom_cfgs=custom_cfgs)
        print('\nTraining started...')
        print(f'Total epochs: {self.num_epochs}')
        print(f'Total episodes: {self.num_epochs * 60}')
        agent.learn()
        print('Training done!')

        try:
            agent.plot(smooth=1)
        except Exception as e:
            print(f'Plotting failed: {e}')


if __name__ == "__main__":
    # Train on medium scenario only with custom parameters
    trainer = TrainerCPO(
        scenario='medium',
        steps_per_episode=18000,      # 36000 steps per episode
        steps_per_epoch= 36000,
        num_epochs=100,              # 100 epochs total
        device="cpu"                 # Use CPU (change to "cuda:0" if GPU available)
    )
    trainer.train(run_id=0)