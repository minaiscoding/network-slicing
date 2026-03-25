from numpy.random import default_rng
from multi_gnb_wrapper import MultiGNBWrapper
from node_b import NodeB

# You must create real slices here
# Example placeholders:
 #gnb1 = NodeB(id=0, x=200, y=200, slices_l1=[...], slots_per_step=50, n_prbs=100, coverage_radius=300)
 #gnb2 = NodeB(id=1, x=700, y=200, slices_l1=[...], slots_per_step=50, n_prbs=100, coverage_radius=300)

env = MultiGNBWrapper(
    gnb_list=[gnb1, gnb2],
    handover_hysteresis=0.03,
    handover_ttt=2,
    verbose=True
)

obs, info = env.reset()
print("obs shape:", obs.shape)
print("info:", info)

ue_id = env.add_ue(x=100, y=200, vx=20, vy=0)

for t in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"t={t}, reward={reward:.3f}, info={info}")
    env.render()