import gymnasium as gym
from Brain import SACAgent
# from Common import Play, Logger, get_params
from Common import get_params, Logger
import numpy as np
from tqdm import tqdm
import mujoco
from mate.agents import GreedyTargetAgent, GreedyCameraAgent
from Brain.agent import RunningMeanStd
from mate.wrappers.rescaled_observation import RescaledObservation
from Common.config_args import *
import mate

# Dieu chinh observation -> local state
# Dieu chinh cai moi truong thanh moi truong discrete 

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()
    base_env = gym.make(params["env_name"], config=ENV_CONFIG)
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    total_dim = OBS_DIM + params["n_skills"]
    normalizer = RunningMeanStd(total_dim)
    n_states = OBS_DIM
    n_actions = ACTION_DIM
    action_bounds = [-2.5, 2.5] # real bound of MATE: [-5, -2.5] -> [5, 2.5]

    obs, _ = env.reset()

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    env.close()
    
    del env, n_states, n_actions, action_bounds

    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            # env.seed(params["seed"])
            # env.observation_space.seed(params["seed"])
            # env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            obs, _ = env.reset()
            obs = obs[0]
            obs = concat_state_latent(obs, z, params["n_skills"]) # create an one-hot vector
            episode_reward = 0
            logq_zses = []

            max_n_steps = params["max_episode_len"]
            for step in range(1, 1 + max_n_steps):
                
                normalizer.update(obs)
                obs = normalizer.normalize(obs)
                
                action = agent.choose_action(obs) # obs for an agent
                actions = np.array([action for _ in range(NUM_AGENTS)])

                next_obs, reward, done, trunc, _ = env.step(actions)
                next_obs = next_obs[0]
                next_obs = concat_state_latent(next_obs, z, params["n_skills"])
                agent.store(obs, z, done, action, next_obs)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                obs = next_obs
                if done:
                    break

            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       np.random.get_state(),
                    #    env.np_random.get_state(),
                    #    env.observation_space.np_random.get_state(),
                    #    env.action_space.np_random.get_state(),
                       *agent.get_rng_states(),
                       )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
