import mate
from mate.agents import GreedyTargetAgent
from util.mate.util_config import _normalize_state
# Base environment for MultiAgentTracking
base_env = mate.make('MultiAgentTracking-v0')
base_env = mate.DiscreteCamera(base_env, levels=5)  # uncomment for discrete setting
env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent(seed=0))
done = False
for i in range(100):
    camera_joint_observation = env.reset()
    print("test = ", camera_joint_observation[0][13:15])
    print("test = ", _normalize_state(camera_joint_observation))
    print("test scale = ", _normalize_state(camera_joint_observation)[0][13:15])
    print("test shape = ", _normalize_state(camera_joint_observation)[0].shape)
    
print(".action_space.n = ", env.action_space)
print(".obseravtion_space = ", env.observation_space)
while not done:
    camera_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
    camera_joint_observation, camera_team_reward, done, camera_infos = env.step(camera_joint_action)
    print("camera_infos = ", camera_infos[0]['coverage_rate'])