import gym, tqdm
import numpy as np
from overcooked_ai_py.utils import mean_and_std_err, merge_dictionaries
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


DEFAULT_ENV_PARAMS = {
    "horizon": 400
}

MAX_HORIZON = 1e10

class OvercookedEnv(object):
    """An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, mdp, start_state_fn=None, horizon=MAX_HORIZON, debug=False):
        """
        mdp (OvercookedGridworld or function): either an instance of the MDP or a function that returns MDP instances 
        start_state_fn (OvercookedState): function that returns start state for the MDP, called at each environment reset
        horizon (float): number of steps before the environment returns done=True
        """
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
            self.variable_mdp = True
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
            self.variable_mdp = False
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")
        
        self.horizon = horizon
        self.start_state_fn = start_state_fn
        self.reset()
        if self.horizon >= MAX_HORIZON and self.state.order_list is None and debug:
            print("Environment has (near-)infinite horizon and no terminal states")

    def __repr__(self):
        """Standard way to view the state of an environment programatically
        is just to print the Env object"""
        return self.mdp.state_string(self.state)

    def print_state_transition(self, a_t, r_t, info):
        print("Timestep: {}\nJoint action taken: {} \t Reward: {} + shape * {} \n{}\n".format(
            self.t, tuple(Action.ACTION_TO_CHAR[a] for a in a_t), r_t, info["shaped_r"], self)
        )

    @property
    def env_params(self):
        return {
            "start_state_fn": self.start_state_fn,
            "horizon": self.horizon
        }

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    @staticmethod
    def print_state(mdp, s):
        e = OvercookedEnv(mdp, s)
        print(e)

    def copy(self):
        return OvercookedEnv(
            mdp=self.mdp.copy(),
            start_state_fn=self.start_state_fn,
            horizon=self.horizon
        )

    def step(self, joint_action):
        """Performs a joint action, updating the environment state
        and providing a reward.
        
        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        next_state, sparse_reward, reward_shaping = self.mdp.get_state_transition(self.state, joint_action)
        self.cumulative_sparse_rewards += sparse_reward
        self.cumulative_shaped_rewards += reward_shaping
        self.state = next_state
        self.t += 1
        done = self.is_done()
        info = {'shaped_r': reward_shaping}
        if done:
            info['episode'] = {
                'ep_sparse_r': self.cumulative_sparse_rewards,
                'ep_shaped_r': self.cumulative_shaped_rewards,
                'ep_length': self.t
            }
        return (next_state, sparse_reward, done, info)

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.mdp = self.mdp_generator_fn()
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

    def is_done(self):
        """Whether the episode is over."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start 
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display: print("Starting state\n{}".format(self))
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display: print(self)
            if done: break
        successor_state = self.state
        self.reset()
        return successor_state, done

    def run_agents(self, agent_pair, include_final_state=False, display=False, display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t, info_t).
        """
        assert self.cumulative_sparse_rewards == self.cumulative_shaped_rewards == 0, \
            "Did not reset environment before running agents"
        trajectory = []
        done = False

        if display: print(self)
        while not done:
            s_t = self.state

            # Getting actions and action infos (optional) for both agents
            joint_action_and_infos = agent_pair.joint_action(s_t)
            a_t, a_info_t = zip(*joint_action_and_infos)
            assert all(a in Action.ALL_ACTIONS for a in a_t)
            assert all(type(a_info) is dict for a_info in a_info_t)

            s_tp1, r_t, done, info = self.step(a_t)
            info["agent_infos"] = a_info_t
            trajectory.append((s_t, a_t, r_t, done, info))

            if display and self.t < display_until:
                self.print_state_transition(a_t, r_t, info)

        assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True, None))

        return np.array(trajectory), self.t, self.cumulative_sparse_rewards, self.cumulative_shaped_rewards

    def get_rollouts(self, agent_pair, num_games, display=False, final_state=False, agent_idx=0, reward_shaping=0.0, display_until=np.Inf, info=True, metadata_fn=lambda x: {}):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed 
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took), 
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format 
        (baselines, stable_baselines, etc)

        metadata_fn returns some metadata information computed at the end of each trajectory based on
        some of the trajectory data.

        NOTE: standard trajectories format used throughout the codebase
        """
        trajectories = {
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [], # Individual (sparse) reward values
            "ep_dones": [], # Individual done values
            "ep_infos": [],

            # With shape (n_episodes, ):
            "ep_returns": [], # Sum of sparse rewards across each episode
            "ep_lengths": [], # Lengths of each episode
            "mdp_params": [], # Custom MDP params to for each episode
            "env_params": [], # Custom Env params for each episode

            # Custom metadata key value pairs
            "metadatas": [] # Final data type is a dictionary of similar format to trajectories
        }

        range_fn = tqdm.trange if info else range
        for i in range_fn(num_games):
            agent_pair.set_mdp(self.mdp)

            rollout_info = self.run_agents(agent_pair, display=display, include_final_state=final_state, display_until=display_until)
            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = rollout_info
            obs, actions, rews, dones, infos = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3], trajectory.T[4]
            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_infos"].append(infos)
            trajectories["ep_returns"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)
            trajectories["metadatas"].append(metadata_fn(rollout_info))

            self.reset()
            agent_pair.reset()

        mu, se = mean_and_std_err(trajectories["ep_returns"])
        if info: print("Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
            mu, np.std(trajectories["ep_returns"]), se, num_games, np.mean(trajectories["ep_lengths"]))
        )

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}

        # Merging all metadata dictionaries, assumes same keys throughout all
        trajectories["metadatas"] = merge_dictionaries(trajectories["metadatas"])

        # TODO: should probably transfer check methods over to Env class
        from overcooked_ai_py.agents.benchmarking import AgentEvaluator
        AgentEvaluator.check_trajectories(trajectories)
        return trajectories

    @staticmethod
    def get_agent_infos_for_trajectories(trajectories, agent_idx):
        """
        Returns a dictionary of the form
        {
            "[agent_info_0]": [ [episode_values], [], ... ],
            "[agent_info_1]": [ [], [], ... ],
            ...
        }
        with as keys the keys returned by the agent in it's agent_info dictionary
        """
        agent_infos = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            ep_infos = trajectories["ep_infos"][traj_idx]
            traj_agent_infos = [step_info["agent_infos"][agent_idx] for step_info in ep_infos]

            # Append all dictionaries together
            traj_agent_infos = merge_dictionaries(traj_agent_infos)
            agent_infos.append(traj_agent_infos)

        # Append all dictionaries together once again
        agent_infos = merge_dictionaries(agent_infos)
        agent_infos = {k: np.array(v) for k, v in agent_infos.items()}
        return agent_infos

    @staticmethod
    def proportion_stuck_time(trajectories, agent_idx, stuck_time=3):
        stuck_matrix = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            stuck_matrix.append([])
            obs = trajectories["ep_observations"][traj_idx]
            for traj_timestep in range(stuck_time, trajectories["ep_lengths"][traj_idx]):
                if traj_timestep >= stuck_time:
                    recent_states = obs[traj_timestep - stuck_time : traj_timestep + 1]
                    recent_player_pos_and_or = [s.players[agent_idx].pos_and_or for s in recent_states]

                    if len({item for item in recent_player_pos_and_or}) == 1:
                        # If there is only one item in the last stuck_time steps, then we classify the agent as stuck
                        stuck_matrix[traj_idx].append(True)
                    else:
                        stuck_matrix[traj_idx].append(False)
                else:
                    stuck_matrix[traj_idx].append(False)
        return stuck_matrix


class MultiOvercookedEnv(object):
    """This stores multiple OvercookedEnvs with identical
    settings (e.g. same mdp). Initially designed so
    that we can simultaneously run several overcooked games.
    """

    def __init__(self, num_envs, mdp, start_state_fn=None, horizon=MAX_HORIZON, debug=False):
        """
        mdp (OvercookedGridworld or function): either an instance of the MDP or a function that returns MDP instances
        start_state_fn (OvercookedState): function that returns start state for the MDP, called at each environment reset
        horizon (float): number of steps before the environment returns done=True
        """
        # Properties shared across all envs:
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
            self.variable_mdp = True
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
            self.variable_mdp = False
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")
        self.horizon = horizon
        self.start_state_fn = start_state_fn
        if self.horizon >= MAX_HORIZON and self.state.order_list is None and debug:
            print("Environment has (near-)infinite horizon and no terminal states")
        self.num_envs = num_envs
        self.mdp = mdp
        self.env_params = {"start_state_fn": self.start_state_fn, "horizon": self.horizon}
        self.envs = [OvercookedEnv(mdp, start_state_fn, horizon, debug) for _ in range(num_envs)]
        self.reset()

    def reset(self):
        """Reset all envs"""
        [self.envs[i].reset() for i in range(self.num_envs)]

    def get_asymm_rollouts(self, asymm_agent_pairs, num_games=1, info=True, metadata_fn=lambda x: {}):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories. NOTE: Only currently set up for num_games = 1

        asymm_agent_pairs: A collection of agents, comprised of a single AgentFromPolicy, and a
        list of N partner agents, e.g. ToMModels and/or ImitationAgentFromPolicyWithHistorys.
        The idea is that each of the N partners is paired with AgentFromPolicy. When running
        get_rollouts, we can send N states to AgentFromPolicy simultaneously; whereas we need
        query the ToMModels separately.

        See OvercookedEnv.get_rollouts for further documentation.
        """

        assert num_games == 1, "Only currently set up for num_games = 1"

        trajectories = [{
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [], # Individual (sparse) reward values
            "ep_dones": [], # Individual done values
            "ep_infos": [],

            # With shape (n_episodes, ):
            "ep_returns": [], # Sum of sparse rewards across each episode
            "ep_lengths": [], # Lengths of each episode
            "mdp_params": [], # Custom MDP params to for each episode
            "env_params": [], # Custom Env params for each episode

            # Custom metadata key value pairs
            "metadatas": [] # Final data type is a dictionary of similar format to trajectories
        } for _ in range(self.num_envs)]

        asymm_agent_pairs.set_mdp(self.mdp)
        rollout_infos = self.run_asymm_agents(asymm_agent_pairs)

        for i in range(self.num_envs):
            rollout_info = rollout_infos[i]
            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = rollout_info
            obs, actions, rews, dones, infos = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3], trajectory.T[4]
            trajectories[i]["ep_observations"].append(obs)
            trajectories[i]["ep_actions"].append(actions)
            trajectories[i]["ep_rewards"].append(rews)
            trajectories[i]["ep_dones"].append(dones)
            trajectories[i]["ep_infos"].append(infos)
            trajectories[i]["ep_returns"].append(tot_rews_sparse)
            trajectories[i]["ep_lengths"].append(time_taken)
            trajectories[i]["mdp_params"].append(self.mdp.mdp_params)
            trajectories[i]["env_params"].append(self.env_params)
            trajectories[i]["metadatas"].append(metadata_fn(rollout_info))

            mu, se = mean_and_std_err(trajectories[i]["ep_returns"])
            if info: print("Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
                mu, np.std(trajectories[i]["ep_returns"]), se, num_games, np.mean(trajectories[i]["ep_lengths"]))
            )

            # Converting to numpy arrays
            trajectories[i] = {k: np.array(v) for k, v in trajectories[i].items()}

            # Merging all metadata dictionaries, assumes same keys throughout all
            trajectories[i]["metadatas"] = merge_dictionaries(trajectories[i]["metadatas"])

            # TODO: should probably transfer check methods over to Env class
            from overcooked_ai_py.agents.benchmarking import AgentEvaluator
            AgentEvaluator.check_trajectories(trajectories[i])

        self.reset()  # Resets all envs
        asymm_agent_pairs.reset()

        return trajectories

    def run_asymm_agents(self, asymm_agent_pairs):
        """
        Trajectories returned are a list of trajectories, one for each pair of agents. Each
        trajectory is a list of state-action pairs (s_t, joint_a_t, r_t, done_t, info_t).
        """
        assert False not in [self.envs[i].cumulative_sparse_rewards == self.envs[i].cumulative_shaped_rewards == 0
                for i in range(self.num_envs)], "Did not reset environment before running agents"

        trajectories = [[] for _ in range(self.num_envs)]
        output = []
        dones = [False for _ in range(self.num_envs)]

        while not dones[0]:

            assert [dones[0] == dones[i] for i in range(self.num_envs)], "Not all envs are done at the same time!"
            states_t = [self.envs[i].state for i in range(self.num_envs)]
            joint_actions_and_infos = asymm_agent_pairs.joint_actions(states_t)

            for i in range(self.num_envs):
                # Process the output for each env:
                a_t, a_info_t = zip(*joint_actions_and_infos[i])
                assert all(a in Action.ALL_ACTIONS for a in a_t)
                assert all(type(a_info) is dict for a_info in a_info_t)

                # Step this env (independently of the other envs):
                s_tp1, r_t, done, info = self.envs[i].step(a_t)
                info["agent_infos"] = a_info_t
                trajectories[i].append((states_t[i], a_t, r_t, done, info))
                dones[i] = done
                if done:
                    output.append([np.array(trajectories[i]), self.envs[i].t,
                              self.envs[i].cumulative_sparse_rewards, self.envs[i].cumulative_shaped_rewards])

        assert [len(trajectories[i]) == self.envs[i].t for i in range(self.num_envs)], \
                    "Trajectory has wrong length"
        return output


class Overcooked(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.

    NOTE: Observations returned are in a dictionary format with various information that is
    necessary to be able to handle the multi-agent nature of the environment. There are probably
    better ways to handle this, but we found this to work with minor modifications to OpenAI Baselines.
    
    NOTE: The index of the main agent in the mdp is randomized at each reset of the environment, and 
    is kept track of by the self.agent_idx attribute. This means that it is necessary to pass on this 
    information in the output to know for which agent index featurizations should be made for other agents.
    
    For example, say one is training A0 paired with A1, and A1 takes a custom state featurization.
    Then in the runner.py loop in OpenAI Baselines, we will get the lossless encodings of the state,
    and the true Overcooked state. When we encode the true state to feed to A1, we also need to know
    what agent index it has in the environment (as encodings will be index dependent).
    """

    def custom_init(self, base_env, featurize_fn, baselines_reproducible=False):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines_reproducible:
            # NOTE: To prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which 
            # seeding does not) reach, we set the same seed internally to all
            # environments. The effect is negligible, as all other randomness
            # is controlled by the actual run seeds
            np.random.seed(0)

        self.base_env = base_env
        self.featurize_fn = featurize_fn
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()

    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape
        high = np.ones(obs_shape) * max(dummy_mdp.soup_cooking_time, dummy_mdp.num_items_for_soup, 5)
        return gym.spaces.Box(high * 0, high, dtype=np.float32)

    def step(self, action):
        """
        action: 
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
        
        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, info = self.base_env.step(joint_action)
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, next_state)
        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        
        obs = {"both_agent_obs": both_agents_ob,
                "overcooked_state": next_state,
                "other_agent_env_idx": 1 - self.agent_idx}
        return obs, reward, done, info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        self.mdp = self.base_env.mdp
        self.agent_idx = np.random.choice([0, 1])
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return {"both_agent_obs": both_agents_ob, 
                "overcooked_state": self.base_env.state, 
                "other_agent_env_idx": 1 - self.agent_idx}

    def render(self, mode='human', close=False):
        pass
