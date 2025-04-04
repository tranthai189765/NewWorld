import mate
import time
import torch
from filter import obstacle_filter as o_f
from filter import target_filter as t_f
from filter import camera_filter as c_f
from filter import env_target_filter as et_f
from filter import env_base_filter as eb_f
from buffer.target_buffer import TargetBuffer as b_t
from buffer.obstacle_buffer import ObstacleBuffer as b_o
from buffer.camera_buffer import CameraBuffer as b_c
from perception import EncodeLinear

env = mate.make('MultiAgentTracking-v0')
env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
env.seed(0)
done = False
buffer = b_t(max_size=5, obs_shape=(8, 4))
dim_in = 20  # Kích thước đầu vào
dim_out = 32  # Kích thước embedding đầu ra

# Tự động chọn CPU hoặc GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncodeLinear(dim_in, dim_out, device=device)

print(f"Model is running on: {device}")  # Kiểm tra thiết bị
camera_joint_observation = env.reset()
for i in range(17):
    env.render()
    camera_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
    camera_joint_observation, camera_team_reward, done, camera_infos = env.step(camera_joint_action)
    target_state = env.get_real_opponent_info()
    print(camera_joint_observation)
    print("i = ", i, "Collected :", eb_f.collected_infos(camera_joint_observation))
    # print(t_f.collected_infos(camera_joint_observation))
    # print(et_f.collected_infos(target_state))
    time.sleep(1000)
    # buffer.store(t_f.collected_infos(camera_joint_observation))

print("buffer")
print(buffer.buffer)
# print("done")
# print(buffer.take())
print("output")
print(buffer.take(3))