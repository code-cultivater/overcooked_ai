import tqdm
import random
import numpy as np
from hr_coordination.mdp.overcooked_mdp import OvercookedGridworld, Action, Direction

class OvercookedEnv(object):
    """An environment containing a single agent that can take actions.

    The environment keeps track of the current state of the agent, and updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, mdp, start_state=None, horizon=float('inf'), random_start_pos=True, random_start_objs=True):
        """
        start_state (OvercookedState): what the environemt resets to when calling reset
        horizon (float): number of steps before the environment returns True to .is_done()
        """
        self.mdp = mdp
        if start_state is not None:
            self.start_state_fn = lambda: start_state
        else:
            self.start_state_fn = lambda: self.mdp.get_start_state(random_start_pos=random_start_pos,
                                                                   random_start_objs=random_start_objs)
        self.horizon = horizon
        self.reset()

    def __repr__(self):
        return self.mdp.state_string(self.state)

    def get_current_state(self):
        return self.state

    def get_actions(self, state):
        return self.mdp.get_actions(state)

    def step(self, joint_action):
        """Performs a joint action, updating the state and providing a reward."""
        assert not self.is_done()
        state = self.get_current_state()
        next_state, reward, shaped_reward, prob = self.get_random_next_state(state, joint_action)
        self.cumulative_rewards += reward
        self.cumulative_shaped_rewards += shaped_reward
        self.state = next_state
        self.t += 1
        done = self.is_done()
        info = {"prob": prob}
        if done:
            info['episode'] = {
                'sparse_r': self.cumulative_rewards,
                'dense_r': self.cumulative_shaped_rewards,
                'l': self.t, 
            }
        info['dense_r'] = shaped_reward
        return (next_state, reward, done, info)

    def get_random_next_state(self, state, action):
        """Chooses the next state according to T(state, action)."""
        results, reward, shaped_reward = self.mdp.get_transition_states_and_probs(state, action)

        # If deterministic, don't generate a random number
        if len(results) == 1:
            return (results[0][0], reward, shaped_reward, 1.0)

        rand = random.random()
        sum = 0.0
        for next_state, prob in results:
            sum += prob
            if sum > 1.0:
                raise ValueError('Total transition probability more than one.')
            if rand < sum:
                return (next_state, reward, shaped_reward, prob)
        raise ValueError('Total transition probability less than one.')

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.state = self.start_state_fn()
        self.cumulative_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

    def is_done(self):
        """Returns True if the episode is over and the agent cannot act."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    @staticmethod
    def execute_plan(mdp, start_state, action_plan, display=False, horizon=np.Inf):
        """Executes action_plan from start_state in mdp and returns resulting state."""
        env = OvercookedEnv(mdp, start_state, horizon=horizon)
        env.state = start_state
        if display: print("Starting state\n{}".format(env))
        for a in action_plan:
            env.step(a)
            if display: print(env)
            if env.is_done():
                break
        successor_state = env.state
        return successor_state, env.is_done()

    def run_agents(self, agent_pair, display=False, displayEnd=False, final_state=False, joint_actions=False):
        """
        Trajectory returned will a list of state-action pairs (s_t, a_t, r_t, d_t), in which
        the last element will be the last state visited and a None joint action.
        Therefore, there will be t + 1 tuples in the trajectory list.
        """
        trajectory = []
        done = False

        if display: print(self)
        while not done:
            s_t = self.state
            a_t = agent_pair.joint_action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, info = self.step(a_t)
            trajectory.append((s_t, a_t, r_t, done))
            
            if display or (done and displayEnd): 
                print("Timestep: {}\nJoint action: {} \t Reward: {} + shape * {} \n{}".
                      format(self.t, a_t, r_t, info["dense_r"], self))

        # Add final state
        # TODO: Clean up
        if final_state:
            trajectory.append((s_tp1, (None, None), 0, True))
            assert len(trajectory) == self.t + 1, "{} vs {}".format(len(trajectory), self.t)
        else:
            assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)
        
        time_taken, tot_rewards, tot_shaped_rewards = self.t, self.cumulative_rewards, self.cumulative_shaped_rewards

        # Reset environment
        self.reset()
        trajectory = np.array(trajectory)
        return trajectory, time_taken, tot_rewards, tot_shaped_rewards

    def get_rollouts(self, agent_pair, num_games, display=False, displayEnd=False, processed=False,
                     final_state=False, agent_idx=0, reward_shaping=0.0):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed 
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took), 
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format 
        (baselines, stable_baselines, etc)
        """
        trajectories = {

            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [], # Individual reward values
            "ep_dones": [], # Individual done values

            # With shape (n_episodes, ):
            "ep_returns": [], # Sum of rewards across each episode
            "ep_lengths": [] # Lengths of each episode

        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews, tot_rews_shaped = self.run_agents(agent_pair, display=display,
                                                                                displayEnd=displayEnd, final_state=final_state)
            obs, actions, rews, dones = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3]  # .T computes transpose
            if processed:
                # NOTE: only actions and observations for agent `agent_idx`
                obs = np.array([self.mdp.preprocess_observation(state)[agent_idx] for state in obs])
                actions = np.array([np.array([Action.ACTION_TO_INDEX[joint_action[agent_idx]]])
                                    for joint_action in actions]).astype(int)

            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews + tot_rews_shaped * reward_shaping)
            trajectories["ep_lengths"].append(time_taken)

        print("Avg reward {} over {} games of avg length {}".format(
            np.mean(trajectories["ep_rewards"]), num_games, np.mean(trajectories["ep_lengths"])))

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        return trajectories

    @staticmethod
    def print_state(mdp, s):
        e = OvercookedEnv(mdp, s)
        print(e)

class VariableOvercookedEnv(OvercookedEnv):
    """Wrapper for Env class which changes mdp at each reset from a mdp_generator function"""

    def __init__(self, mdp_generator_fn, horizon=float('inf')):
        """
        start_state (OvercookedState): what the environemt resets to when calling reset
        horizon (float): number of steps before the environment returns True to .is_done()
        """
        self.mdp_generator_fn = mdp_generator_fn
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.mdp = self.mdp_generator_fn()
        self.start_state = self.mdp.get_start_state()
        self.state = self.start_state
        self.cumulative_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

import gym
from gym import spaces
class Overcooked(gym.Env):
    """Wrapper for the Env class above that is compatible with gym API
    
    The convention is that all processed observations returned are ordered in such a way
    to be standard input for the main agent policy. The index of the main agent in the mdp 
    is randomized at each reset of the environment, and is kept track of by the self.agent_idx
    attribute.
    
    One can use the switch_player function to change the observation to be in standard 
    format for the secondary agent policy.
    """

    def custom_init(self, base_env, joint_actions=False, featurize_fn=None):
        """
        base_env_fn: a function that when called will return a initialized version of the
                     Env class. Can be called again to reset the environment.
        """
        self.base_env = base_env
        self.joint_actions = joint_actions

        dummy_state = self.base_env.mdp.get_start_state()

        if featurize_fn is None:
            self.featurize_fn = self.base_env.mdp.preprocess_observation
            obs_shape = self.base_env.mdp.preprocess_observation(dummy_state)[0].shape
            high = np.ones(obs_shape) * 5
        else:
            self.featurize_fn = featurize_fn
            obs_shape = featurize_fn(dummy_state)[0].shape
            high = np.ones(obs_shape) * 10 # NOTE: arbitrary right now

        self.observation_space = spaces.Box(high * 0, high, dtype=np.float32)

        if self.joint_actions:
            self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS)**2)
        else:
            self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()

    def step(self, action):
        """
        action: 
            (self.agent_idx action, other agent action)
            is a tuple with the action of the primary and secondary agents in index format
        
        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        if self.joint_actions:
            action = Action.INDEX_TO_ACTION_INDEX_PAIRS[action]
        else:
            assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, info = self.base_env.step(joint_action)
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return both_agents_ob, reward, done, info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.
        """
        # If fixed map, reset it, otherwise, generate new one at each reset
        self.base_env.reset()
        self.agent_idx = np.random.choice([0, 1])
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return both_agents_ob

    def render(self, mode='human', close=False):
        pass

##########################
# RLLIB Env (deprecated) #
##########################
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# class OvercookedMultiAgent(MultiAgentEnv):

#     def __init__(self, **env_config):
#         self.overcooked = Overcooked(env_config)
#         self.agent_names = ["agent0", "agent1"]

#     @property
#     def observation_space(self):
#         return self.overcooked.observation_space

#     @property
#     def action_space(self):
#         return self.overcooked.action_space

#     def reset(self):
#         obs = self.overcooked.reset()
#         other_obs = OvercookedGridworld.switch_player(obs)
#         return {"agent0": obs, "agent1": other_obs}

#     def step(self, action_dict):
#         assert all(self.overcooked.action_space.contains(a) for a in action_dict.values()), "%r (%s) invalid"%(action_dict)
        
#         agent_action, other_agent_action = [Action.INDEX_TO_ACTION[action_dict[a]] for a in self.agent_names]

#         joint_action = (agent_action, Direction.STAY)
#         next_state, reward, done, info = self.overcooked.base_env.step(joint_action)

#         mdp = self.overcooked.mdp
#         observation0 = mdp.preprocess_observation(next_state, mdp, primary_agent_idx=0)
#         observation1 = mdp.preprocess_observation(next_state, mdp, primary_agent_idx=1)

#         obs, rewards, dones, infos = {}, {}, {}, {}
        
#         obs[self.agent_names[0]] = observation0
#         obs[self.agent_names[1]] = observation1
#         for k in action_dict.keys():
#             rewards[k] = reward
#             dones[k] = done
#             infos[k] = info

#         if all(dones.values()):
#             dones["__all__"] = True
#         else:
#             dones["__all__"] = False

#         return obs, rewards, dones, infos