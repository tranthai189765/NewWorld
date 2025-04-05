import torch
import time
import mate
from mate.agents import GreedyCameraAgent, GreedyTargetAgent
import os
import numpy as np
from filter import camera_filter as c_f
from filter import obstacle_filter as o_f
from filter import target_filter as t_f
from filter import env_target_filter as et_f
from buffer.target_buffer import TargetBuffer as b_t
from buffer.obstacle_buffer import ObstacleBuffer as b_o
from buffer.camera_buffer import CameraBuffer as b_c

MAX_EPISODE_STEPS = 4000
MAX_EPISODES = 500
num_dataset = 0
def main():
    num_label = 0
    global num_dataset  # Biến đếm dataset để lưu file
    SEED = 42  # Chọn một giá trị seed cố định để đảm bảo reproducibility
    np.random.seed(SEED)

    for episode in range(MAX_EPISODES):
        base_env = mate.make('MATE-4v4-0-v0')
        base_env = mate.RenderCommunication(base_env)
        env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent(seed=SEED + episode))
        env.reset(seed=SEED + episode)  # Đặt seed cho môi trường nếu hỗ trợ

        buffer_obstacles = b_o(max_size=30, obs_shape=(0, 3))
        buffer_targets = b_t(max_size=30, obs_shape=(4, 4))
        buffer_cameras = b_c(max_size=30, obs_shape=(4, 9))

        camera_agents = GreedyCameraAgent().spawn(env.num_cameras)
        camera_joint_observation = env.reset()
        mate.group_reset(camera_agents, camera_joint_observation)
        camera_infos = None

        inputs_list = []  # Lưu input
        labels_list = []  # Lưu label

        for i in range(MAX_EPISODE_STEPS):
            #print("i = ", i)
            camera_joint_action = mate.group_step(
                env, camera_agents, camera_joint_observation, camera_infos
            )

            results = env.step(camera_joint_action)
            camera_joint_observation, camera_team_reward, done, camera_infos = results
            buffer_cameras.store(c_f.collected_infos(camera_joint_observation))
            buffer_obstacles.store(o_f.collected_infos(camera_joint_observation))
            buffer_targets.store(t_f.collected_infos(camera_joint_observation))
            target_state = env.get_real_opponent_info()

            if i % 40 == 6:
                pending_data = {
                    "cameras": buffer_cameras.take(5),
                    "obstacles": buffer_obstacles.take(),
                    "targets": buffer_targets.take(5),
                }

            if i % 40 == 16:
                label = et_f.collected_infos(target_state)
                labels_list.append(label)
                inputs_list.append(pending_data)
                #print("input = ", pending_data)
                pending_data = None
                #print("label = ", label)
                label = None 
                num_label += 1
                print("numdata = ", num_label)

            # env.render()
            if done:
                break

        # Lưu dữ liệu sau mỗi episode
        save_data(inputs_list, labels_list)

def save_data(inputs, labels):
    global num_dataset
    save_path = f"dataset_{num_dataset}.pt"
    
    #print("inputs = ", inputs)
    #print("labels = ", labels)
    #print("num_dataset = ", num_dataset)

        # Chuyển đổi từng thành phần của inputs thành tensor
    input_tensors = {
        "cameras": torch.tensor([inp["cameras"] for inp in inputs], dtype=torch.float32),
        "obstacles": torch.tensor([inp["obstacles"] for inp in inputs], dtype=torch.float32),
        "targets": torch.tensor([inp["targets"] for inp in inputs], dtype=torch.float32),
    }
    
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    # Lưu vào file .pt
    data_dict = {
        "inputs": input_tensors,
        "labels": label_tensor
    }

    torch.save(data_dict, save_path)
    print(f"Saved dataset: {save_path}")
    num_dataset += 1  # Tăng ID file dataset để tránh ghi đè


if __name__ == '__main__':
    main()
