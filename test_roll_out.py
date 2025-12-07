import copy
import numpy as np
import mate
from mate.agents import GreedyTargetAgent

def one_step_rollout(env, action):
    """Rollout 1 step trên bản copy của env và trả về reward."""
    env_clone = copy.deepcopy(env)
    obs, reward, done, info = env_clone.step(action)
    return reward, obs, done, info


def choose_best_action(env):
    """
    Trả về action tốt nhất khi thử tất cả actions trên env hiện tại.
    """
    n_actions = env.action_space.n
    action_rewards = []

    for a in range(n_actions):
        reward, obs, done, info = one_step_rollout(env, a)
        # nếu reward là list (multi-agent), bạn có thể sum hoặc lấy reward của agent nào đó
        if isinstance(reward, list):
            rew_value = sum(reward)     # hoặc reward[0], tùy bạn
        else:
            rew_value = reward
        action_rewards.append(rew_value)

    best_action = int(np.argmax(action_rewards))
    return best_action, action_rewards

if __name__ == "__main__":
    base_env = mate.make('MultiAgentTracking-v0')
    base_env = mate.DiscreteCamera(base_env, levels=5)  # uncomment for discrete setting
    env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent(seed=0))
    print("env.action_space = ", env.action_space)
    obs = env.reset()

    best_action, all_rewards = choose_best_action(env)
    print("All rewards:", all_rewards)
    print("Best action =", best_action)
