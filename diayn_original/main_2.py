import gym
from .Brain import SACAgentDiscrete
from .Common import Logger, get_params
import numpy as np
# from tqdm import tqdm
# import mujoco_py
import mate
from mate.agents import GreedyTargetAgent
from .Common.config_args import *
from torch.utils.tensorboard import SummaryWriter

def concat_state_latent(s, z_, n):
    # print("s[0].shape = ", s[0].shape)
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    # print("z_one_hot = ", z_one_hot.shape)
    return np.concatenate([s[0], z_one_hot])

def _normalize_state(state):
    #print("raw_state = ", state)
    n_c = int(state[0][0])
    n_t = int(state[0][1])
    n_o = int(state[0][2])
    for camera_index in range(n_c):
        val = int(state[camera_index][3])
        mapping = {
            0: [0, 0, 0, 1],
            1: [0, 0, 1, 0],
            2: [0, 1, 0, 0],
            3: [1, 0, 0, 0]
        }
        original_radius = state[camera_index][19]
        encoded_bits = mapping.get(val, [0,0,0,0])  # fallback nếu giá trị ngoài [0,3]
        state[camera_index][0:4] = np.array(encoded_bits, dtype=float)
        state[camera_index][4:16] = state[camera_index][4:16]/1000.0
        state[camera_index][16:18] = state[camera_index][16:18]/state[camera_index][19]
        state[camera_index][18] = state[camera_index][18]/180.0
        state[camera_index][19] = 1.0
        state[camera_index][20:22] = state[camera_index][20:22]/180.0
        for target_index in range(n_t):
            state[camera_index][22 + 5 * target_index : 22 + 5 * target_index + 3] = state[camera_index][22 + 5 * target_index : 22 + 5 * target_index + 3]/1000.0
        for obstacle_index in range(n_o):
            state[camera_index][22+5*n_t+4*obstacle_index : 22+5*n_t+4*obstacle_index+3]  = state[camera_index][22+5*n_t+4*obstacle_index : 22+5*n_t+4*obstacle_index+3]/1000.0
        for tm_index in range(n_c):
            state[camera_index][22+5*n_t+4*n_o+7*tm_index: 22+5*n_t+4*n_o+7*tm_index+3] = state[camera_index][22+5*n_t+4*n_o+7*tm_index: 22+5*n_t+4*n_o+7*tm_index+3]/1000.0
            state[camera_index][22+5*n_t+4*n_o+7*tm_index+3: 22+5*n_t+4*n_o+7*tm_index+5] = state[camera_index][22+5*n_t+4*n_o+7*tm_index+3: 22+5*n_t+4*n_o+7*tm_index+5]/original_radius
            state[camera_index][22+5*n_t+4*n_o+7*tm_index+5] = state[camera_index][22+5*n_t+4*n_o+7*tm_index+5]/180.0

    return state

if __name__ == "__main__":
    params = get_params()
    
    n_states = STATE_DIM
    n_actions = NUM_ACTIONS
    n_observations = OBS_DIM
    action_bounds = None  

    params.update({"n_local_states": n_states,
                   "n_actions": n_actions,
                   "n_states": n_observations,
                   "action_bounds": action_bounds,
                   'do_train': True,
                   'env_name': 'mate489'})

    print("params:", params)
    del n_states, n_actions, n_observations, action_bounds

    base_env = mate.make('MultiAgentTracking-v0')
    base_env = mate.DiscreteCamera(base_env, levels=5)  
    env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent(seed=0))

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgentDiscrete(p_z=p_z, **params)
    logger = Logger(agent, **params)

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir="./runs/sac_discrete")

    if params["do_train"]:
        if not params.get("train_from_scratch", True):
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
            env.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        global_step = 0

        for episode in range(1 + min_episode, params["max_n_episodes"] + 1):
            print(f"[LOG] Start episode {episode}: ")
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = _normalize_state(state)
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0

            max_n_steps = params["max_episode_len"]
            for step in range(1, 1 + max_n_steps):
                action_agent = agent.choose_action(state)
                action = np.full(NUM_AGENTS, action_agent)  
                next_state, reward, done, camera_infos = env.step(action)
                # camera_joint_observation, camera_team_reward, done, camera_infos
                next_state = _normalize_state(next_state)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action_agent, next_state)

                # --- Train agent and get losses ---
                train_result = agent.train()  # should return q_loss, policy_loss, disc_loss, reward
                if train_result is not None:
                    q_loss, policy_loss, disc_loss, mean_reward = train_result
                    # log to tensorboard
                    tb_writer.add_scalar("Loss/Q_loss", q_loss, global_step)
                    tb_writer.add_scalar("Loss/Policy_loss", policy_loss, global_step)
                    tb_writer.add_scalar("Loss/Discriminator_loss", -disc_loss, global_step)
                    tb_writer.add_scalar("Reward/mean_reward", mean_reward, global_step)

                episode_reward += camera_infos[0]['coverage_rate']
                state = next_state
                global_step += 1
                if done:
                    break

            logger.log(
                episode,
                episode_reward,
                z,
                0.0,                         
                step,                        
                np.random.get_state(),
                env.np_random.get_state(),
                env.observation_space.np_random.get_state(),
                env.action_space.np_random.get_state(),
                *agent.get_rng_states(),
            )
            
    tb_writer.close()