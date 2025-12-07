import copy
import numpy as np
from env.executor import ExecutorEnv
import torch

def one_step_rollout(env, agent_idx, action, goal):
    """
    Rollout 1-step: agent_idx thực hiện action,
    các agent còn lại thực hiện action = 0.
    """
    env_clone = copy.deepcopy(env)

    N = env_clone.n_agents
    actions = np.zeros(N, dtype=int)   # tất cả agent = 0
    actions[agent_idx] = action        # agent cần test = action

    obs, reward, done = env_clone.step(goal, actions)
    print("actions = ", actions)

    # reward shape: [N, 1]
    return reward[agent_idx], obs, done


def choose_best_action(env, agent_idx, goal):
    """
    Trả về action tốt nhất cho agent_idx trên state hiện tại.
    """
    n_actions = env.num_skills
    action_rewards = []

    for a in range(n_actions):
        rew, obs, done = one_step_rollout(env, agent_idx, a, goal)
        action_rewards.append(rew)

    best_action = int(np.argmax(action_rewards))
    return best_action, action_rewards


if __name__ == "__main__":
    env = ExecutorEnv(
        skill_file="./Checkpoints/mate/2025-12-03-21-58-03/params.pth",
        num_steps=10,
        num_skills=10,
        seed=123
    )

    obs = env.reset()

    agent_idx = 0              
    goal = torch.tensor([1,0,0,0,0,0,0,1]).repeat(4, 1)
    print(goal.shape)

    best_action, all_rewards = choose_best_action(env, agent_idx, goal)

    print("Rewards of all actions:", all_rewards)
    print("Best action:", best_action)