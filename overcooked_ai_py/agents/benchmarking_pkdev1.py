import os
import json
import tqdm
import numpy as np
from argparse import ArgumentParser

from hr_coordination.utils import load_dict_from_file, get_max_iter, save_pickle, load_pickle
from hr_coordination.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from hr_coordination.mdp.layout_generator import LayoutGenerator
from hr_coordination.agents.agent import AgentPair, CoupledPlanningAgent, RandomAgent, GreedyHumanModel
from hr_coordination.mdp.overcooked_mdp import OvercookedGridworld, Action
from hr_coordination.mdp.overcooked_env import OvercookedEnv


class AgentEvaluator(object):
    """
    Class used to get trajectory rollouts of agents trained with a variety of methods
    """

    def __init__(self, layout_name, order_goal=['any'], explosion_time=500, start_state=None, horizon=2000, force_compute=False):
        self.layout_name = layout_name
        self.order_goal = order_goal
        self.explosion_time = explosion_time
        self.start_state = start_state
        self.horizon = horizon
        self.force_compute = force_compute
        self._mlp = None
        self._mdp = None
        self._env = None

    @staticmethod
    def from_config(config, start_state=None):
        ae = AgentEvaluator(
            layout_name=config["FIXED_MDP"], 
            order_goal=config["ORDER_GOAL"],
            explosion_time=config["EXPLOSION_TIME"],
            start_state=start_state,
            horizon=config["ENV_HORIZON"]
        )
        from hr_coordination.pbt.pbt_utils import setup_mdp_env
        ae._env = setup_mdp_env(display=False, **config)
        ae._mdp = ae._env.mdp
        ae.config = config
        return ae

    @staticmethod
    def from_pbt_dir(run_dir, start_state=None):
        from hr_coordination.pbt.pbt_utils import setup_mdp_env, get_config_from_pbt_dir
        config = get_config_from_pbt_dir(run_dir)
        return AgentEvaluator.from_config(config, start_state)

    @property
    def mdp(self):
        if self._mdp is None:
            print("Computing Mdp")
            self._mdp = OvercookedGridworld.from_file("data/layouts/" + self.layout_name + ".layout", self.order_goal, self.explosion_time, rew_shaping_params=None)
        return self._mdp

    @property
    def env(self):
        if self._env is None:
            print("Computing Env")
            self._env = OvercookedEnv(self.mdp, start_state=self.start_state, horizon=self.horizon, random_start_objs=False, random_start_pos=False)
        return self._env

    @property
    def mlp(self):
        if self._mlp is None:
            print("Computing Planner")
            self._mlp = MediumLevelPlanner.from_pickle_or_compute("data/planners/" + self.layout_name + "_am.pkl", self.mdp, NO_COUNTERS_PARAMS, force_compute=self.force_compute)
        return self._mlp

    def evaluate_human_model_pair(self, display=True):
        a0 = GreedyHumanModel(self.mlp, player_index=0)
        a1 = GreedyHumanModel(self.mlp, player_index=1)
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair)

    def evaluate_optimal_pair(self, display=True):
        a0 = CoupledPlanningAgent(self.mlp, player_index=0)
        a1 = CoupledPlanningAgent(self.mlp, player_index=1)
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair)

    def evaluate_one_optimal_one_random(self):
        a0 = CoupledPlanningAgent(self.mlp, player_index=0)
        a1 = RandomAgent()
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair)

    def evaluate_one_optimal_one_greedy_human(self, h_idx=0, display=True):
        h, r = GreedyHumanModel, CoupledPlanningAgent
        if h_idx == 0:
            a0, a1 = h(self.mlp, player_index=0), r(self.mlp, player_index=1)
        elif h_idx == 1:
            a0, a1 = r(self.mlp, player_index=0), h(self.mlp, player_index=1)
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair)

    def evaluate_agent_pair(self, agent_pair, num_games=1, display=True):
        agent_pair.set_mdp(self.mdp)
        return self.env.get_rollouts(agent_pair, num_games, display=display)

    def get_pbt_agent_from_path(self, agent_idx=0):
        from hr_coordination.pbt.pbt_utils import get_agent_from_saved_model
        assert self.config, "Class instance has to be initialized with from_pbt_dir"
        agent_folder = self.config["SAVE_DIR"] + 'agent{}'.format(agent_idx)
        agent0_to_load_path = agent_folder  + "/pbt_iter" + str(get_max_iter(agent_folder))
        agent0 = get_agent_from_saved_model(agent0_to_load_path, self.config["SIM_THREADS"])
        return agent0

    def get_pbt_agents_trajectories(self, agent0_idx, agent1_idx, num_trajectories, display=False):
        from hr_coordination.pbt.pbt_utils import setup_mdp_env
        agent0 = self.get_pbt_agent_from_path(agent0_idx)
        agent1 = self.get_pbt_agent_from_path(agent1_idx)

        mdp_env = setup_mdp_env(display=False, **self.config)
        return mdp_env.get_rollouts(AgentPair(agent0, agent1), num_trajectories, display=display, processed=True, final_state=False)

    @staticmethod
    def cumulative_rewards_from_trajectory(trajectory):
        cumulative_rew = 0
        for trajectory_item in trajectory:
            r_t = trajectory_item[2]
            cumulative_rew += r_t
        return cumulative_rew

    def dump_action_trajectory_as_json(self, trajectory, path):
        """
        Trajectory will be a list of state-action pairs (s_t, a_t, r_t).
        Used to visualize trajectories with graphics.
        """
        # Add trajectory to json
        traj = []
        for item in trajectory:
            s_t, a_t, r_t = item
            a_modified = [a if a != Action.INTERACT else a.upper() for a in a_t]
            if all([a is not None for a in a_t]):
                traj.append(a_modified)

        json_traj = {}
        json_traj["traj"] = traj

        # Add layout grid to json
        mdp_grid = []
        for row in self.mdp.terrain_mtx:
            mdp_grid.append("".join(row))

        for i, start_pos in enumerate(self.mdp.start_player_positions):
            x, y = start_pos
            row_string = mdp_grid[y]
            new_row_string = row_string[:x] + str(i + 1) + row_string[x+1:]
            mdp_grid[y] = new_row_string

        json_traj["mdp_grid"] = mdp_grid

        with open(path + '.json', 'w') as filename:  
            json.dump(json_traj, filename)

    @staticmethod
    def save_traj_in_baselines_format(rollout_trajs, filename):
        """Useful for GAIL and behavioral cloning"""
        np.savez(
            filename,
            obs=rollout_trajs["ep_observations"],
            acs=rollout_trajs["ep_actions"],
            ep_lens=rollout_trajs["ep_lengths"],
            ep_rets=rollout_trajs["ep_returns"],
        )
    
    @staticmethod
    def save_traj_in_stable_baselines_format(rollout_trajs, filename):
        stable_baselines_trajs_dict = {
            'actions': np.concatenate(rollout_trajs["ep_actions"]),
            'obs': np.concatenate(rollout_trajs["ep_observations"]),
            'rewards': np.concatenate(rollout_trajs["ep_rewards"]),
            'episode_starts': np.concatenate(rollout_trajs["ep_dones"]),
            'episode_returns': rollout_trajs["ep_returns"]
        }
        stable_baselines_trajs_dict = { k:np.array(v) for k, v in stable_baselines_trajs_dict.items() }
        np.savez(filename, **stable_baselines_trajs_dict)

    # Clean this if unnecessary
    # trajectory, time_taken = self.env.run_agents(agent_pair, display=display)
    # tot_rewards = self.cumulative_rewards_from_trajectory(trajectory)
    # return tot_rewards, time_taken, trajectory

    # @staticmethod
    # def save_state_trajectory_as_pickle(ep_states, ep_joint_actions, ep_lengths, ep_rews, ep_metadata, filename):
    #     data = {"states": states, "joint_actions": joint_actions,  "metadata": metadata}
    #     save_pickle(data, filename)

    # @staticmethod
    # def load_state_trajectory_from_pickle(filename):
    #     data = load_pickle(filename)
    #     return data["states"], data["joint_actions"], data["metadata"]



# SAMPLE SCRIPTS    

# Getting Trajs From Optimal Planner
# eva = AgentEvaluator("scenario2")
# tot_rewards, time_taken, trajectory = eva.evaluate_optimal_pair(["any"] * 3)
# eva.dump_trajectory_as_json(trajectory, "../overcooked-js/simple_rr")
# print("done")

# Getting Trajs from pbt Agent
# eva = AgentEvaluator.from_pbt_dir(run_dir="2019_03_20-10_53_03_scenario2_no_rnd_objs", seed_idx=0)
# ep_rews, ep_lens, ep_obs, ep_acts = eva.get_pbt_agents_trajectories(agent0_idx=0, agent1_idx=0, num_trajectories=1)
# eva.dump_trajectory_as_json(trajectory, "data/agent_runs/")


# if __name__ == "__main__" :
#     parser = ArgumentParser()
#     parser.add_argument("-t", "--type", dest="type",
#                         help="type of run: ['rollouts', 'ppo']", required=True)
#     parser.add_argument("-r", "--run_name", dest="run",
#                         help="name of run in data/*_runs/", required=True)
#     parser.add_argument("-a", "--agent_num", dest="agent_num", default=0)
#     parser.add_argument("-i", "--idx", dest="idx", default=0)

#     args = parser.parse_args()

#     run_type, run_name, player_idx, agent_num = args.type, args.run, int(args.idx), int(args.agent_num)
