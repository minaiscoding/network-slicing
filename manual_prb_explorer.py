#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual PRB Allocation Explorer with KPI Display and Violation Rollback

Allows interactive PRB allocation with:
- Manual PRB input for each slice
- KPI metrics display (delay, loss, throughput, etc.)
- Automatic rollback on SLA violation
- Continuous exploration with state persistence
"""

import numpy as np
import sys
from scenario_creator import create_env


class ManualPRBExplorer:
    def __init__(self, scenario_idx=0, n_steps=1000, slot_length=1e-3):
        """
        Initialize the explorer
        
        Args:
            scenario_idx: Which scenario to use (0-4)
            n_steps: Max environment steps
            slot_length: Slot duration in seconds
        """
        self.scenario_idx = scenario_idx
        self.n_steps = n_steps
        self.slot_length = slot_length
        
        # Create environment
        rng = np.random.RandomState(42)
        self.env = create_env(rng, n=scenario_idx, L1_level=True)
        
        # Track state history
        self.step_count = 0
        self.state_history = []
        self.reward_history = []
        self.violation_count = 0
        self.successful_actions = 0
        
        # Get number of slices (from unwrapped environment)
        node_b = self.env.unwrapped.node_b
        self.n_slices = len(node_b.slices_l1)
        
        print("\n" + "="*80)
        print("MANUAL PRB ALLOCATION EXPLORER")
        print("="*80)
        print("Scenario: {}".format(scenario_idx))
        print("Number of slices: {}".format(self.n_slices))
        print("Total PRBs available: {}".format(node_b.n_prbs))
        print("Slot duration: {:.1f}ms".format(self.slot_length*1000))
        print("="*80)
    
    def display_kpis(self, full_info, reward, action=None, violated=False):
        """Display KPI metrics from slice info"""
        print("\n" + "-"*80)
        if violated:
            print("STEP {} | Reward: {:+.2f} | STATUS: *** VIOLATION ***".format(self.step_count, reward))
        else:
            print("STEP {} | Reward: {:+.2f} | STATUS: OK".format(self.step_count, reward))
        print("-"*80)
        
        if action is not None:
            print("Action (PRB allocation): {}".format(action))
            print()
        
        # full_info has structure: {'l1_info': [slice_info_list], 'violations': ..., 'SLA_labels': ...}
        l1_infos = full_info.get('l1_info', [])
        
        for l1_idx, slice_info_dict in enumerate(l1_infos):
            print("L1 Slice {}:".format(l1_idx))
            
            # slice_info_dict has structure like {0: {...}, 1: {...}} where keys are slice IDs
            for slice_id in sorted(slice_info_dict.keys()):
                metrics = slice_info_dict[slice_id]
                print("  Slice {} Info:".format(slice_id))
                
                # Traffic metrics
                if 'cbr_traffic' in metrics:
                    print("    CBR Traffic:        {:>10.0f} bits".format(metrics['cbr_traffic']))
                    print("    CBR Throughput:     {:>10.0f} bps".format(metrics['cbr_th']))
                    print("    CBR PRBs used:      {:>10.1f}".format(metrics['cbr_prb']))
                    print("    CBR Queue:          {:>10.0f} bits".format(metrics['cbr_queue']))
                    print("    CBR SNR:            {:>10.1f} dB".format(metrics['cbr_snr']))
                
                # Delay metrics (eMBB: mean, URLLC: max)
                if 'cbr_delay_mean' in metrics:
                    print("    CBR Delay (mean):   {:>10.2f} ms <<<< DELAY".format(metrics['cbr_delay_mean']))
                    print("    CBR Pkt Loss:       {:>10.4f}".format(metrics['cbr_packet_loss']))
                elif 'cbr_delay_max' in metrics:
                    print("    CBR Delay (max):    {:>10.2f} ms <<<< DELAY".format(metrics['cbr_delay_max']))
                    print("    CBR Pkt Loss:       {:>10.4f}".format(metrics['cbr_packet_loss']))
                
                if 'vbr_traffic' in metrics:
                    print("    VBR Traffic:        {:>10.0f} bits".format(metrics['vbr_traffic']))
                    print("    VBR Throughput:     {:>10.0f} bps".format(metrics['vbr_th']))
                    print("    VBR PRBs used:      {:>10.1f}".format(metrics['vbr_prb']))
                    print("    VBR Queue:          {:>10.0f} bits".format(metrics['vbr_queue']))
                    print("    VBR SNR:            {:>10.1f} dB".format(metrics['vbr_snr']))
                
                if 'vbr_delay_mean' in metrics:
                    print("    VBR Delay (mean):   {:>10.2f} ms <<<< DELAY".format(metrics['vbr_delay_mean']))
                    print("    VBR Pkt Loss:       {:>10.4f}".format(metrics['vbr_packet_loss']))
                elif 'vbr_delay_max' in metrics:
                    print("    VBR Delay (max):    {:>10.2f} ms <<<< DELAY".format(metrics['vbr_delay_max']))
                    print("    VBR Pkt Loss:       {:>10.4f}".format(metrics['vbr_packet_loss']))
                
                # mMTC specific
                if 'devices' in metrics:
                    print("    Devices:            {:>10.1f}".format(metrics['devices']))
                    print("    Avg Repetitions:    {:>10.1f}".format(metrics['avg_rep']))
                    print("    Delay:              {:>10.2f} ms".format(metrics['delay']))
                
                print()
    
    def get_prb_allocation(self):
        """Get PRB allocation from user input"""
        node_b = self.env.unwrapped.node_b
        print("\n" + "="*80)
        print("ENTER PRB ALLOCATION FOR STEP {}".format(self.step_count + 1))
        print("="*80)
        print("Total PRBs available: {}".format(node_b.n_prbs))
        print("Number of slices: {}".format(self.n_slices))
        print("Enter PRBs as comma-separated numbers")
        suggested = ",".join([str(node_b.n_prbs // self.n_slices)] * self.n_slices)
        print("Example: {}".format(suggested))
        print("\nCommands: 'exit' to quit, 'reset' to reset scenario")
        print("-"*80)
        
        while True:
            try:
                user_input = input("\n>>> Enter PRBs: ").strip().lower()
                
                if user_input == 'exit':
                    return None
                elif user_input == 'reset':
                    return 'reset'
                
                # Parse input
                prbs_str = user_input.split(',')
                prbs = [int(x.strip()) for x in prbs_str]
                
                if len(prbs) != self.n_slices:
                    print("ERROR: Expected {} values, got {}".format(self.n_slices, len(prbs)))
                    print("Try again or type 'exit' to quit")
                    continue
                
                total = sum(prbs)
                if total != node_b.n_prbs:
                    print("ERROR: Total PRBs = {} but must equal {}".format(total, node_b.n_prbs))
                    print("Try again or type 'exit' to quit")
                    continue
                
                if any(p < 0 for p in prbs):
                    print("ERROR: PRB values cannot be negative")
                    print("Try again or type 'exit' to quit")
                    continue
                
                return np.array(prbs, dtype=np.float32)
            
            except ValueError as e:
                print("PARSE ERROR: {}".format(e))
                print("Enter only comma-separated numbers like: 50,30,20")
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                return None
    
    def step_with_violation_check(self, action):
        """
        Execute step and check for violations
        
        Returns: (did_violate, reward, state, info)
        """
        # Execute action
        state, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Get info from NodeB
        node_b = self.env.unwrapped.node_b
        slice_info = node_b.get_info()
        
        # Check for violation (negative reward = SLA violated)
        if reward < 0:
            self.violation_count += 1
            return True, reward, state, slice_info
        else:
            self.state_history.append(state.copy())
            self.reward_history.append(reward)
            self.successful_actions += 1
            return False, reward, state, slice_info
    
    def run(self):
        """Main exploration loop"""
        print("\nStarting manual PRB exploration...")
        print("Each step:")
        print("  1. You enter PRB allocation")
        print("  2. System shows KPI metrics")
        print("  3. If SLA violated: re-enter different allocation")
        print("  4. If SLA met: continue to next step")
        print()
        
        while self.step_count < self.n_steps:
            # Get user input
            action = self.get_prb_allocation()
            
            if action is None:
                break
            elif isinstance(action, str) and action == 'reset':
                # Reset environment
                rng = np.random.RandomState(42)
                self.env = create_env(rng, n=self.scenario_idx, L1_level=True)
                self.step_count = 0
                self.state_history = []
                self.reward_history = []
                self.violation_count = 0
                self.successful_actions = 0
                print("\n>>> Environment reset to initial state\n")
                continue
            
            # Execute step and check for violations
            did_violate, reward, state, info = self.step_with_violation_check(action)
            
            # Display KPIs
            self.display_kpis(info, reward, action, violated=did_violate)
            
            if did_violate:
                print("*** SLA VIOLATION DETECTED ***")
                print("*** Reward: {:.2f} (negative = violation) ***".format(reward))
                print("*** You must re-enter a different PRB allocation ***")
                print("*** The environment state will REVERT to before this action ***")
                print()
                
                # Decrement step count since this was a violation
                self.step_count -= 1
                continue
            else:
                print(">>> Action ACCEPTED! Proceeding to next step...")
                print()
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print session summary"""
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        print("Total steps completed: {}".format(self.step_count))
        print("Successful actions: {}".format(self.successful_actions))
        print("SLA violations encountered: {}".format(self.violation_count))
        if self.reward_history:
            print("Average reward (successful steps): {:.3f}".format(np.mean(self.reward_history)))
            print("Min reward: {:.3f}".format(np.min(self.reward_history)))
            print("Max reward: {:.3f}".format(np.max(self.reward_history)))
        print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual PRB Allocation Explorer with KPI Display and Violation Rollback")
    parser.add_argument('--scenario', type=int, default=0, 
                       help='Scenario index (0-4, default: 0)')
    parser.add_argument('--steps', type=int, default=50,
                       help='Max number of steps (default: 50)')
    
    args = parser.parse_args()
    
    # Validate scenario
    if args.scenario < 0 or args.scenario > 4:
        print("ERROR: Scenario must be 0-4")
        print("  0: eMBB (200 PRBs, 5 users)")
        print("  1: eMBB+mMTC (150 PRBs, 3 eMBB + 2 mMTC)")
        print("  2: eMBB+mMTC (100 PRBs, 1 eMBB + 4 mMTC)")
        print("  3: eMBB+mMTC (100 PRBs, 1 eMBB + 4 mMTC)")
        print("  4: URLLC (70 PRBs, 1 URLLC + 1 mMTC)")
        sys.exit(1)
    
    try:
        explorer = ManualPRBExplorer(scenario_idx=args.scenario, n_steps=args.steps)
        explorer.run()
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
        sys.exit(0)
    except Exception as e:
        print("\n\n[ERROR] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
