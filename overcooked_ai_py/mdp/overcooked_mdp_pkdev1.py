import tqdm
import random
import itertools
import numpy as np
from collections import defaultdict
from hr_coordination.utils import pos_distance

eps = 10e-2

NO_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 0,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0.015,
    "POT_DISTANCE_REW": 0.03,
    "SOUP_DISTANCE_REW": 0.1,
}

class ObjectState(object):
    """State of an object in the Overcooked gridworld.

    name: The name of the object.
    state: Some information about the object, dependent on the name. For
        example, the state of a dish could be "full" or "empty".
    position: (x, y) tuple denoting the location of the object.
    """

    SOUP_TYPES = ['onion', 'tomato']

    def __init__(self, name, position, state=None):
        assert type(position) == tuple
        # TODO: Use numbers instead of strings for name, and have a dictionary
        # to convert to and from
        self.name = name
        self.position = position
        if name == 'soup':
            assert len(state) == 3
        self.state = state

    def is_valid(self):
        if self.name == 'onion':
            return self.state is None
        elif self.name == 'tomato':
            return self.state is None
        elif self.name == 'dish': 
            return self.state is None
        elif self.name == 'soup':
            soup_type, num_items, cook_time = self.state
            valid_soup_type = soup_type in self.SOUP_TYPES
            valid_item_num = (1 <= num_items <= 3)
            valid_cook_time = (0 <= cook_time)
            return valid_soup_type and valid_item_num and valid_cook_time
        else:
            raise ValueError("Unrecognized object")
        return False

    def deepcopy(self):
        return ObjectState(self.name, self.position, self.state)

    def __eq__(self, other):
        return isinstance(other, ObjectState) and \
            self.name == other.name and \
            self.position == other.position and \
            self.state == other.state

    def __hash__(self):
        return hash((self.name, self.position, self.state))

    def __repr__(self):
        if self.state is None:
            return '{}@{}'.format(self.name, self.position)
        return '{}@{} with state {}'.format(
            self.name, self.position, str(self.state))


class PlayerState(object):
    """State of a player in the Overcooked gridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
        None if there is no such object.
    """

    def __init__(self, position, orientation, held_object=None):
        assert type(position) == tuple
        assert orientation in Direction.ALL_DIRECTIONS
        if held_object is not None:
            assert isinstance(held_object, ObjectState)
            assert held_object.position == position

        self.position = position
        self.orientation = orientation
        self.held_object = held_object
        # self.stats = stats if stats is not None else {
        #     "picked_up_onions": 0,
        #     "picked_up_tomatoes": 0,
        #     "placed_in_pot_onions": 0,
        #     "placed_in_pot_tomatoes": 0,
        #     "picked_up_dishes": 0,
        #     "soups_delivered": 0
        # }

    @property
    def pos_and_or(self):
        return (self.position, self.orientation)

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj

    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj

    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(self.position, self.orientation, new_obj)

    def __eq__(self, other):
        return isinstance(other, PlayerState) and \
               self.position == other.position and \
               self.orientation == other.orientation and \
               self.held_object == other.held_object

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return '{} facing {} holding {}'.format(
            self.position, self.orientation, str(self.held_object))

class OvercookedState(object):
    def __init__(self, players, objects, order_list, pot_explosion=False):
        """Represents a state in Overcooked. 

        players: List of PlayerStates.
        objects: Dictionary mapping positions (x, y) to ObjectStates. Does NOT
            include objects held by players.

        Order is important for players but not for objects.
        """
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        assert all([o in OvercookedGridworld.ORDER_TYPES for o in order_list])
        self.order_list = order_list
        self.pot_explosion = pot_explosion

    @property
    def player_positions(self):
        return tuple([player.position for player in self.players])

    @property
    def player_orientations(self):
        return tuple([player.orientation for player in self.players])

    @property
    def players_pos_and_or(self):
        """Returns a ((pos1, or1), (pos2, or2)) tuple"""
        return tuple(zip(*[self.player_positions, self.player_orientations]))

    @property
    def objects_by_type(self):
        objects_by_type = defaultdict(list)
        for pos, obj in self.objects.items():
            objects_by_type[obj.name].append(obj)
        return objects_by_type

    @property
    def player_objects(self):
        player_objects = defaultdict(list)
        for player in self.players:
            if player.has_object():
                player_obj = player.get_object()
                player_objects[player_obj.name].append(player_obj)
        return player_objects

    def has_object(self, pos):
        return pos in self.objects

    def get_object(self, pos):
        assert self.has_object(pos)
        return self.objects[pos]

    def add_object(self, obj, pos=None):
        if pos is None:
            pos = obj.position

        assert not self.has_object(pos)
        obj.position = pos
        self.objects[pos] = obj

    def remove_object(self, pos):
        assert self.has_object(pos)
        obj = self.objects[pos]
        del self.objects[pos]
        return obj

    @staticmethod
    def from_players_pos_and_or(players_pos_and_or, order_list):
        return OvercookedState(
            [PlayerState(*player_pos_and_or) for player_pos_and_or in players_pos_and_or], 
            objects={}, order_list=order_list)

    @staticmethod
    def from_player_positions(player_positions, order_list):
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return OvercookedState.from_players_pos_and_or(dummy_pos_and_or, order_list)

    def deepcopy(self):
        return OvercookedState(
            [player.deepcopy() for player in self.players],
            {pos:obj.deepcopy() for pos, obj in self.objects.items()}, 
            list(self.order_list),
            self.pot_explosion)

    def __eq__(self, other):
        return isinstance(other, OvercookedState) and \
            self.players == other.players and \
            set(self.objects.items()) == set(other.objects.items()) and \
            np.array_equal(self.order_list, other.order_list) and \
            self.pot_explosion == other.pot_explosion

    def __hash__(self):
        return hash((self.players, tuple(self.objects.values()), tuple(self.order_list), self.pot_explosion))

    def __str__(self):
        return 'Players: {}, Objects: {}, Order list: {}, Pot explosion: {}'.format(
            str(self.players), str(list(self.objects.values())), str(self.order_list), str(self.pot_explosion))

class OvercookedGridworld(object):
    """An MDP grid world based off of the Overcooked game."""
    # TODO: Separate into outside params dict
    COOK_TIME = 20
    DELIVERY_REWARD = 20
    POT_EXPLOSION_PENALTY = 0 #TODO: decide if we want a penalty (i.e. -100)
    ORDER_TYPES = ObjectState.SOUP_TYPES + ['any']

    def __init__(self, terrain, player_positions, start_order_list, explosion_time, rew_shaping_params=None, layout_name="unknown_layout"):
        assert explosion_time > 0
        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = player_positions
        self.start_order_list = start_order_list
        self.explosion_time = explosion_time
        self.reward_shaping = BASE_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        # TODO: Add layout name
        self.layout_name = layout_name

    def get_start_state(self, random_start_pos=False, random_start_objs=0.0):
        """Returns the start state."""
        if random_start_pos:
            valid_positions = self.get_valid_joint_player_positions()
            start_pos = valid_positions[np.random.choice(len(valid_positions))]
        else:
            start_pos = self.start_player_positions

        s = OvercookedState.from_player_positions(start_pos, order_list=self.start_order_list)
        thresh = random_start_objs
        if thresh > 0:
            pots = self.get_pot_states(s)["empty"]
            for pot_loc in pots:
                p = np.random.rand()
                if p < thresh:
                    n = int(np.random.randint(low=1, high=4))
                    s.objects[pot_loc] = ObjectState("soup", pot_loc, ('onion', n, 0))

            for player in s.players:
                p = np.random.rand()
                if p < thresh:
                    obj = np.random.choice(["dish", "onion", "soup"], p=[0.2, 0.6, 0.2])
                    if obj == "soup":
                        player.set_object(ObjectState(obj, player.position, ('onion', 3, OvercookedGridworld.COOK_TIME)))
                    else:
                        player.set_object(ObjectState(obj, player.position))
        return s

    def get_actions(self, state):
        """Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.

        Note that you can request moves into terrain, which are equivalent to
        STAY. The order in which actions are returned is guaranteed to be
        deterministic, in order to allow agents to implement deterministic
        behavior.
        """
        self._check_valid_state(state)
        return [self._get_player_actions(state, i) for i in range(len(state.players))]

    def _get_player_actions(self, state, player_num):
        return Action.ALL_ACTIONS

    def _check_action(self, state, joint_action):
        for p_action, p_legal_actions in zip(joint_action, self.get_actions(state)):
            if p_action not in p_legal_actions:
                raise ValueError('Invalid action')

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        return len(state.order_list) == 0 or state.pot_explosion

    def get_transition_states_and_probs(self, state, joint_action):
        """Gets information about possible transitions for the action.

        Returns list of (next_state, prob) pairs representing the states
        reachable from 'state' by taking 'action' along with their transition
        probabilities.
        """
        assert len(state.order_list) != 0, "Trying to find successor of a terminal state: {}".format(state)
        action_sets = self.get_actions(state)
        for player, action, action_set in zip(state.players, joint_action, action_sets):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))

        new_state = state.deepcopy()

        # Resolve interacts first
        sparse_reward, dense_reward = self.resolve_interacts(new_state, joint_action)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations
        
        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        sparse_reward += self.step_environment_effects(new_state)

        # Additional dense reward logic
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]
        dishes_in_play = len(new_state.player_objects['dish'])
        for player_old, player_new in zip(state.players, new_state.players):
            # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
            max_dist = 8

            if player_new.held_object is not None and player_new.held_object.name == 'dish' and len(nearly_ready_pots) >= dishes_in_play:
                min_dist_to_pot_new = np.inf
                min_dist_to_pot_old = np.inf
                for pot in nearly_ready_pots:
                    new_dist = np.linalg.norm(np.array(pot) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(pot) - np.array(player.position))
                    if new_dist < min_dist_to_pot_new:
                        min_dist_to_pot_new = new_dist
                    if old_dist < min_dist_to_pot_old:
                        min_dist_to_pot_old = old_dist
                if min_dist_to_pot_old > min_dist_to_pot_new:
                    dense_reward += self.reward_shaping["POT_DISTANCE_REW"] * (1 - min(min_dist_to_pot_new / max_dist, 1))

            if player_new.held_object is None and len(cooking_pots) > 0 and dishes_in_play == 0:
                min_dist_to_d_new = np.inf
                min_dist_to_d_old = np.inf
                for serving_loc in self.terrain_pos_dict['D']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player.position))
                    if new_dist < min_dist_to_d_new:
                        min_dist_to_d_new = new_dist
                    if old_dist < min_dist_to_d_old:
                        min_dist_to_d_old = old_dist

                if min_dist_to_d_old > min_dist_to_d_new:
                    dense_reward += self.reward_shaping["DISH_DISP_DISTANCE_REW"] * (1 - min(min_dist_to_d_new / max_dist, 1))

            if player_new.held_object is not None and player_new.held_object.name == 'soup':
                min_dist_to_s_new = np.inf
                min_dist_to_s_old = np.inf
                for serving_loc in self.terrain_pos_dict['S']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player.position))
                    if new_dist < min_dist_to_s_new:
                        min_dist_to_s_new = new_dist

                    if old_dist < min_dist_to_s_old:
                        min_dist_to_s_old = old_dist
                
                if min_dist_to_s_old > min_dist_to_s_new:
                    dense_reward += self.reward_shaping["SOUP_DISTANCE_REW"] * (1 - min(min_dist_to_s_new / max_dist, 1))

        return [(new_state, 1.0)], sparse_reward, dense_reward

    def resolve_interacts(self, new_state, joint_action):
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]

        # NOTE: Currently if two players both interact with a terrain, we
        # resolve player 1's interact first and then player 2's, without doing
        # anything like collision checking.
        sparse_reward, dense_reward = 0, 0
        for player, action in zip(new_state.players, joint_action):
            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Direction.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at(i_pos)

            if terrain_type == 'X':
                if player.has_object() and not new_state.has_object(i_pos):
                    new_state.add_object(player.remove_object(), i_pos)
                elif not player.has_object() and new_state.has_object(i_pos):
                    player.set_object(new_state.remove_object(i_pos))

            elif terrain_type == 'O' and player.held_object is None:
                player.set_object(ObjectState('onion', pos))
            elif terrain_type == 'T' and player.held_object is None:
                player.set_object(ObjectState('tomato', pos))
            elif terrain_type == 'D' and player.held_object is None:
                dishes_already = len(new_state.player_objects['dish'])
                player.set_object(ObjectState('dish', pos))

                dishes_on_counters = self.get_counter_objects_dict(new_state)["dish"]
                if len(nearly_ready_pots) > dishes_already and len(dishes_on_counters) == 0:
                    dense_reward += self.reward_shaping["DISH_PICKUP_REWARD"]

            elif terrain_type == 'P' and player.has_object():
                if player.get_object().name == 'dish' and new_state.has_object(i_pos):
                    obj = new_state.get_object(i_pos)
                    assert obj.name == 'soup', 'Object in pot was not soup'
                    _, num_items, cook_time = obj.state
                    if num_items == 3 and cook_time >= OvercookedGridworld.COOK_TIME:
                        player.remove_object()  # Turn the dish into the soup
                        player.set_object(new_state.remove_object(i_pos))
                        dense_reward += self.reward_shaping["SOUP_PICKUP_REWARD"]

                elif player.get_object().name in ['onion', 'tomato']:
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Pot was empty
                        player.remove_object()
                        new_state.add_object(ObjectState('soup', i_pos, (item_type, 1, 0)), i_pos)
                        dense_reward += self.reward_shaping["PLACEMENT_IN_POT_REW"]

                    else:
                        # Pot has already items in it
                        obj = new_state.get_object(i_pos)
                        assert obj.name == 'soup', 'Object in pot was not soup'
                        soup_type, num_items, cook_time = obj.state
                        if num_items < 3 and soup_type == item_type:
                            player.remove_object()
                            obj.state = (soup_type, num_items + 1, 0)
                            dense_reward += self.reward_shaping["PLACEMENT_IN_POT_REW"]

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'soup':
                    soup_type, num_items, cook_time = obj.state
                    assert soup_type in ObjectState.SOUP_TYPES
                    assert num_items == 3 and \
                           cook_time >= OvercookedGridworld.COOK_TIME and \
                           cook_time < self.explosion_time
                    player.remove_object()

                    # If the delivered soup is the one currently required
                    if len(new_state.order_list) == 0:
                        print("Something went wrong!")
                        print("New state", new_state)
                        print(player, action)
                        break
                    current_order = new_state.order_list[0]
                    if current_order == 'any' or soup_type == current_order:
                        new_state.order_list = new_state.order_list[1:]
                        sparse_reward += OvercookedGridworld.DELIVERY_REWARD
                    # If last soup necessary was delivered, stop resolving interacts
                    if len(new_state.order_list) == 0:
                        break

        return sparse_reward, dense_reward

    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        new_positions, new_orientations = self.compute_new_positions_and_orientations(state.players, joint_action)
        for player_state, new_pos, new_o in zip(state.players, new_positions, new_orientations):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(self, old_player_states, joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(zip(*[
            self._move_if_direction(p.position, p.orientation, a) \
            for p, a in zip(old_player_states, joint_action)]))
        old_positions = tuple(p.position for p in old_player_states)
        new_positions = self._handle_collisions(old_positions, new_positions)
        return new_positions, new_orientations

    def _handle_collisions(self, old_positions, new_positions):
        # Assume there are only two players
        if self.is_collision(old_positions, new_positions):
            return old_positions
        return new_positions

    def is_collision(self, old_positions, new_positions):
        p1_old, p2_old = old_positions
        p1_new, p2_new = new_positions
        if p1_new == p2_new:
            return True
        elif p1_new == p2_old and p1_old == p2_new:
            return True
        return False

    def step_environment_effects(self, state):
        reward = 0
        for obj in state.objects.values():
            if obj.name == 'soup':
                x, y = obj.position
                soup_type, num_items, cook_time = obj.state
                if self.terrain_mtx[y][x] == 'P' and cook_time < self.explosion_time and num_items == 3:
                    obj.state = soup_type, num_items, cook_time + 1
                
                if obj.state[2] == self.explosion_time and num_items == 3:
                    state.pot_explosion = True
                    reward = self.POT_EXPLOSION_PENALTY
        return reward
         
    def get_terrain_type_at(self, pos):
        x, y = pos
        return self.terrain_mtx[y][x]

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])

    def get_onion_dispenser_locations(self):
        return list(self.terrain_pos_dict['O'])

    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict['T'])

    def get_serving_locations(self):
        return list(self.terrain_pos_dict['S'])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict['P'])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict['X'])

    def get_pot_states(self, state):
        pots_states_dict = {}
        pots_states_dict['empty'] = []
        pots_states_dict['onion'] = defaultdict(list)
        pots_states_dict['tomato'] = defaultdict(list)
        for pot_pos in self.get_pot_locations():
            if not state.has_object(pot_pos):
                pots_states_dict['empty'].append(pot_pos)
            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if num_items == 3:
                    if cook_time >= OvercookedGridworld.COOK_TIME:
                        pots_states_dict[soup_type]['ready'].append(pot_pos)
                    else:
                        pots_states_dict[soup_type]['cooking'].append(pot_pos)
                elif 0 < num_items < 3:
                    pots_states_dict[soup_type]['partially_full'].append(pot_pos)
                else:
                    raise ValueError("Pot with more than 3 items")
        return pots_states_dict

    def get_counter_objects_dict(self, state, counter_subset=None):
        """Returns a dictionary of pos:objects on counters by type"""
        counters_considered = self.terrain_pos_dict['X'] if counter_subset is None else counter_subset
        counter_objects_dict = defaultdict(list)
        for obj in state.objects.values():
            if obj.position in counters_considered:
                counter_objects_dict[obj.name].append(obj.position)
        return counter_objects_dict

    def get_empty_counters(self, state):
        counter_locations = self.terrain_pos_dict['X']
        return [pos for pos in counter_locations if not state.has_object(pos)]

    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict

    def _move_if_direction(self, position, orientation, action):
        """Returns new position after executing action"""
        if action not in Action.MOTION_ACTIONS:
            return position, orientation
        new_pos = Direction.move_in_direction(position, action)
        new_orientation = orientation if action == Direction.STAY else action
        if new_pos not in self.get_valid_player_positions():
            return position, new_orientation
        return new_pos, new_orientation
    
    def get_valid_player_positions(self):
        return self.terrain_pos_dict[' ']

    def get_valid_joint_player_positions(self):
        valid_joint = []
        valid_positions = self.get_valid_player_positions()
        for pos1, pos2 in itertools.product(valid_positions, valid_positions):
            if pos1 != pos2:
                valid_joint.append((pos1, pos2))
        return valid_joint
                
    def get_valid_player_positions_and_orientations(self):
        valid_states = []
        for pos in self.get_valid_player_positions():
            valid_states.extend([(pos, d) for d in Direction.CARDINAL])
        return valid_states

    def get_valid_joint_player_positions_and_orientations(self):
        """All joint player position and orientation pairs that are not
        overlapping and on empty terrain."""
        valid_player_states = self.get_valid_player_positions_and_orientations()

        valid_joint_player_states = []
        for pos_and_or_1, pos_and_or_2 in itertools.product(valid_player_states, repeat=2):
            p1_pos, p2_pos = pos_and_or_1[0], pos_and_or_2[0]
            if p1_pos != p2_pos:
                valid_joint_player_states.append((pos_and_or_1, pos_and_or_2))
        return valid_joint_player_states

    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for pstate in state.players:
            # Check that players are not on terrain
            pos = pstate.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if pstate.held_object is not None:
                all_objects.append(pstate.held_object)
                assert pstate.held_object.position == pstate.position

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at(obj_pos) != ' '

        # Check that players and non-held objects don't overlap
        all_pos = [pstate.position for pstate in state.players]
        all_pos += [ostate.position for ostate in state.objects.values()]
        assert len(all_pos) == len(set(all_pos)), "Overlapping players or objects"

        # Check that objects have a valid state
        for ostate in all_objects:
            assert ostate.is_valid()
 
    @staticmethod
    def from_file(filename, start_order_list, explosion_time, rew_shaping_params=None):
        with open(filename, 'r') as f:
            return OvercookedGridworld.from_grid(
                f.read().strip().split('\n'),
                start_order_list, explosion_time, rew_shaping_params
            )

    @staticmethod
    def from_grid(grid, start_order_list, explosion_time, rew_shaping_params=None):
        grid = [[c for c in row] for row in grid]
        OvercookedGridworld._assert_valid_grid(grid)

        player_positions = [None, None]
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c in ['1', '2']:
                    grid[y][x] = ' '
                    assert player_positions[int(c) - 1] is None, 'Duplicate player in grid'
                    player_positions[int(c) - 1] = (x, y)

        assert all(position is not None for position in player_positions), 'A player was missing'
        return OvercookedGridworld(grid, player_positions, start_order_list, explosion_time, rew_shaping_params)

    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    grid_string += Direction.ORIENTATION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string += player_object.name[:1]
                    else:
                        grid_string += str(0) if player.position == state.players[0].position else str(1)
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string = grid_string + element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        if soup_type == "onion":
                            grid_string += "ø"
                        elif soup_type == "tomato":
                            grid_string += "†"
                        else:
                            raise ValueError()

                        if num_items == 3:
                            grid_string += str(cook_time)
                        elif num_items == 2:
                            grid_string += "="
                        else:
                            grid_string += "-"
                    else:
                        grid_string += element + " "

            grid_string += "\n"
        grid_string += "Current orders: {}\n".format(state.order_list)

        if state.pot_explosion:
            grid_string += "\nPot exploded! GAME OVER"

        return grid_string

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must not be free spaces
        def is_not_free(c):
            return c in 'XOPDST'

        for y in range(height):
            assert is_not_free(grid[y][0]), 'Left border must not be free'
            assert is_not_free(grid[y][-1]), 'Right border must not be free'
        for x in range(width):
            assert is_not_free(grid[0][x]), 'Top border must not be free'
            assert is_not_free(grid[-1][x]), 'Bottom border must not be free'

        all_elements = [element for row in grid for element in row]
        assert all(c in 'XOPDST12 ' for c in all_elements), 'Invalid character in grid'
        assert all_elements.count('1') == 1, "'1' must be present exactly once"
        assert all_elements.count('2') == 1, "'2' must be present exactly once"
        assert all_elements.count('D') >= 1, "'D' must be present at least once"
        assert all_elements.count('S') >= 1, "'S' must be present at least once"
        assert all_elements.count('P') >= 1, "'P' must be present at least once"
        assert all_elements.count('O') >= 1 or all_elements.count('T') >= 1, "'O' or 'T' must be present at least once"

    def featurize_state(self, overcooked_state, primary_agent_idx, mlp):
        all_features = {}

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = { o_name:idx for idx, o_name in enumerate(IDX_TO_OBJ) }

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            obj = player.held_object
            
            if obj is None:
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                obj_idx = OBJ_TO_IDX[obj.name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest onion
            # TODO: Consider on counters too, say zero if already have an onion/dish/etc
            onion_locations = self.get_onion_dispenser_locations() + counter_objects["onion"]
            all_features["p{}_closest_onion".format(i)] = self.get_deltas_to_closest_location(player, onion_locations, mlp)

            empty_pot_locations = pot_state["empty"]
            all_features["p{}_closest_empty_pot".format(i)] = self.get_deltas_to_closest_location(player, empty_pot_locations, mlp)
            partial_pot_locations = pot_state["onion"]["partially_full"]
            all_features["p{}_closest_partial_pot".format(i)] = self.get_deltas_to_closest_location(player, partial_pot_locations, mlp)
            cooking_pot_locations = pot_state["onion"]["cooking"]
            all_features["p{}_closest_cooking_pot".format(i)] = self.get_deltas_to_closest_location(player, cooking_pot_locations, mlp)
            ready_pot_locations = pot_state["onion"]["ready"]
            all_features["p{}_closest_ready_pot".format(i)] = self.get_deltas_to_closest_location(player, ready_pot_locations, mlp)
            
            all_features["p{}_closest_dish".format(i)] = self.get_deltas_to_closest_location(player, self.get_dish_dispenser_locations(), mlp)
            all_features["p{}_closest_serving".format(i)] = self.get_deltas_to_closest_location(player, self.get_serving_locations(), mlp)

        # TODO: wall info

        features_np = { k:np.array(v) for k, v in all_features.items() }
        
        p0, p1 = overcooked_state.players
        p0_dict = { k:v for k,v in features_np.items() if k[:2] == "p0" }
        p1_dict = { k:v for k,v in features_np.items() if k[:2] == "p1" }
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        ordered_features_p0 = np.concatenate([p0_features, p1_features, p1_rel_to_p0])

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        ordered_features_p1 = np.concatenate([p1_features, p0_features, p0_rel_to_p1])
        return ordered_features_p0, ordered_features_p1

    def get_deltas_to_closest_location(self, player, locations, mlp):
        _, closest_loc = mlp.mp.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
        if closest_loc is None:
            # TODO: think about what this entails
            return (0, 0)
        dy_loc, dx_loc = pos_distance(closest_loc, player.position)
        return dy_loc, dx_loc

    def preprocess_observation(self, overcooked_state, debug=False):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        """"PK: Function that takes an overcooked state and makes an 'observation' to be used by the CNN"""
        assert type(debug) is bool
        base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc", "dish_disp_loc", "serve_loc"]

        # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
        primary_agent_idx = 0
        other_agent_idx = 1 - primary_agent_idx
        ordered_player_features = ["player_{}_loc".format(primary_agent_idx), "player_{}_loc".format(other_agent_idx)] + \
                    ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                    for i, d in itertools.product([primary_agent_idx, other_agent_idx], Direction.CARDINAL)]

        variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc", 
                                "tomatoes_in_pot", "tomatoes_cook_time", "tomato_soup_loc", 
                                "dishes", "onions", "tomatoes"]

        LAYERS = ordered_player_features + base_map_features + variable_map_features
        all_objects = list(overcooked_state.objects.values())
        state_mask_dict = {k:np.zeros(self.shape) for k in LAYERS}

        def make_layer(position, value):
            layer = np.zeros(self.shape)
            layer[position] = value
            return layer

        # MAP LAYERS
        for loc in self.get_counter_locations():
            state_mask_dict["counter_loc"][loc] = 1

        for loc in self.get_pot_locations():
            state_mask_dict["pot_loc"][loc] = 1

        for loc in self.get_onion_dispenser_locations():
            state_mask_dict["onion_disp_loc"][loc] = 1
        
        for loc in self.get_tomato_dispenser_locations():
            state_mask_dict["tomato_disp_loc"][loc] = 1

        for loc in self.get_dish_dispenser_locations():
            state_mask_dict["dish_disp_loc"][loc] = 1

        for loc in self.get_serving_locations():
            state_mask_dict["serve_loc"][loc] = 1

        # PLAYER LAYERS
        for i, player in enumerate(overcooked_state.players):
            player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            state_mask_dict["player_{}_loc".format(i)] = make_layer(player.position, 1)
            state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(
                player.position, 1)
            obj = player.held_object
            if obj is not None:
                all_objects.append(obj)


        # OBJECT & STATE LAYERS
        for obj in all_objects:
            if obj.name == "soup":
                soup_type, num_onions, cook_time = obj.state
                if soup_type == "onion":
                    if obj.position in self.get_pot_locations():
                        soup_type, num_onions, cook_time = obj.state
                        state_mask_dict["onions_in_pot"] += make_layer(obj.position, num_onions)
                        state_mask_dict["onions_cook_time"] += make_layer(obj.position, cook_time)
                    else:
                        # If player soup is not in a pot, put it in separate mask
                        state_mask_dict["onion_soup_loc"] += make_layer(obj.position, 1)
                elif soup_type == "tomato":
                    if obj.position in self.get_pot_locations():
                        soup_type, num_tomatoes, cook_time = obj.state
                        state_mask_dict["tomatoes_in_pot"] += make_layer(obj.position, num_onions)
                        state_mask_dict["tomatoes_cook_time"] += cook_time
                    else:
                        # If player soup is not in a pot, put it in separate mask
                        state_mask_dict["tomato_soup_loc"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized soup")

            elif obj.name == "dish":
                state_mask_dict["dishes"] += make_layer(obj.position, 1)
            elif obj.name == "onion":
                state_mask_dict["onions"] += make_layer(obj.position, 1)
            elif obj.name == "tomato":
                state_mask_dict["tomatoes"] += make_layer(obj.position, 1)
            else:
                raise ValueError("Unrecognized object")

        if debug:
            print(len(LAYERS))
            print(len(state_mask_dict))
            for k, v in state_mask_dict.items():
                print(k)
                print(np.transpose(v, (1, 0)))

        # Stack of all the state masks, order decided by order of LAYERS
        state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
        state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
        assert state_mask_stack.shape[:2] == self.shape
        assert state_mask_stack.shape[2] == len(LAYERS)
        # TODO: Include orders/time left!!!
        final_obs_p0 = np.array(state_mask_stack).astype(int)
        final_obs_p1 = OvercookedGridworld.switch_player(final_obs_p0)

        # Check that the func to turn the obs back into a state is working:
        self.check_state_to_obs_to_state(final_obs_p0, player_observing=0, state=overcooked_state)
        self.check_state_to_obs_to_state(final_obs_p1, player_observing=1, state=overcooked_state)

        return final_obs_p0, final_obs_p1

    def check_state_to_obs_to_state(self, observation, player_observing, state):

        state_from_obs_from_state = self.state_from_observation(observation, player_observing)
        # print(state_from_obs_from_state)
        if state.players != state_from_obs_from_state.players or state.objects != state_from_obs_from_state.objects:
            # It's fine if they're not equal due to the info lost about the number of onions in cooked soup:
            # This could be a held cooked-soup:
            for i in range(2):
                if state_from_obs_from_state.players[i].has_object() and \
                        state_from_obs_from_state.players[i].held_object.name == 'soup':
                    # Take the 'cooking time' value from the actual state... then the states should be the same
                    state_from_obs_from_state.players[i].held_object.state = (
                        state_from_obs_from_state.players[i].held_object.state[0],
                        state_from_obs_from_state.players[i].held_object.state[1],
                        state.players[i].held_object.state[2])
            # Or a cooked-soup on the counter:
            for loc in state_from_obs_from_state.objects:
                if state_from_obs_from_state.objects[loc].name == 'soup' and loc not in self.get_pot_locations():
                    # Take the 'cooking time' value from the actual state... then the states should be the same
                    state_from_obs_from_state.objects[loc].state = (state_from_obs_from_state.objects[loc].state[0],
                                                                    state_from_obs_from_state.objects[loc].state[1],
                                                                    state.objects[loc].state[2])
        assert state.players == state_from_obs_from_state.players
        assert state.objects == state_from_obs_from_state.objects



    def state_from_observation(self, observation, player_observing):
        """
        We want to reverse the function preprocess_observation, which takes a state and produces an
        observation (the observation is what is fed into the NN model for DRL agents).
        :player_observing: the index of the player observing"""

        base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc", "dish_disp_loc", "serve_loc"]

        # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
        primary_agent_idx = player_observing
        other_agent_idx = 1 - primary_agent_idx

        ordered_player_features = ["player_{}_loc".format(primary_agent_idx), "player_{}_loc".format(other_agent_idx)] + \
                                  ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                                   for i, d in itertools.product([primary_agent_idx, other_agent_idx], Direction.CARDINAL)]
        variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc",
                                 "tomatoes_in_pot", "tomatoes_cook_time", "tomato_soup_loc",
                                 "dishes", "onions", "tomatoes"]
        LAYERS = ordered_player_features + base_map_features + variable_map_features
        state_mask_dict = {k: observation[:,:,j] for j, k in enumerate(LAYERS)}


        # Order list: Just set to Any/onions all the way down?
        order_list = ["any"] * 10

        # Extract positions and values from a layer:
        def extract_pos_value(layer):
            #TODO: Must be more efficient way of doing this (e.g. use np.nonzero? I tried this but it's hard to get the
            # data types right, and the indices are the wrong way round!)
            positions = []
            values = []
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    if layer[(i, j)] != 0:
                        positions.append((i, j))
                        values.append(layer[(i, j)])
            return positions, values

        # players: List of PlayerStates
        # A player state is defined by 'position', 'orientation' and 'held_object'. So for each player we just need to
        # supply these, then do player = PlayerState(position, orientation, held_object) to make the player state

        players = []
        for i in range(2):  # Assuming 2 players
            # Player position:
            layer_player_pos = state_mask_dict["player_{}_loc".format(i)]
            player_pos, _ = extract_pos_value(layer_player_pos)
            player_pos = player_pos[0]

            # Player orientation:
            for j in range(Direction.CARDINAL.__len__()):
                player_orientation_idx = j
                layer_temp = state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)]
                if not np.array_equal(layer_temp, np.zeros(self.shape)):
                    # If this orientation layer isn't all zeros:
                    player_or = Direction.INDEX_TO_DIRECTION[player_orientation_idx]

            # Player objects held:
            layers_to_search_for_held_objects = ["onion_soup_loc", "dishes", "onions"]
            possible_held_objects = ["soup", "dish", "onion"]
            number_held_objects = 0
            held_object_state = None
            for j, object in enumerate(layers_to_search_for_held_objects):
                layer_temp = state_mask_dict[object]
                if layer_temp[player_pos] == 1:
                    # If this layer has a 1 where the player is then the player is holding the object
                    held_object_name = possible_held_objects[j]
                    number_held_objects += 1
                    if held_object_name == 'soup':
                        #TODO: Assuming onion
                        soup_type = 'onion'
                        num_onions = 3  # This info isn't in the observation -- presumably it's 3!
                        cook_time = 99  # This info isn't in the observation
                        soup_state = (soup_type, num_onions, cook_time)
                        held_object_state = ObjectState(held_object_name, player_pos, soup_state)
                    else:
                        held_object_state = ObjectState(held_object_name, player_pos)
            assert number_held_objects <= 1

            # Now create the player state:
            players.append(PlayerState(player_pos, player_or, held_object_state))


        # Objects: Dictionary mapping positions (x, y) to ObjectStates (not inc objects held by players)

        objects_dict = {}

        # Consider each layer separately:

        # Onion:
        positions, _ = extract_pos_value(state_mask_dict['onions'])
        for i in range(positions.__len__()):  # For each position containing an onion
            if positions[i] != players[0].position and positions[i] != players[1].position:  # If it's held by a player then it's
                # part of the player's state
                object_name = 'onion'
                object_position = positions[i]
                objects_dict[object_position] = ObjectState(object_name, object_position)

        # Dish:
        positions, _ = extract_pos_value(state_mask_dict['dishes'])
        for i in range(positions.__len__()):  # For each position containing an onion
            if positions[i] != players[0].position and positions[i] != players[1].position:  # If it's held by a player then it's
                # part of the player's state
                object_name = 'dish'
                object_position = positions[i]
                objects_dict[object_position] = ObjectState(object_name, object_position)

        # Soup outside of pot:
        positions, _ = extract_pos_value(state_mask_dict['onion_soup_loc'])
        for i in range(positions.__len__()):  # For each position containing an onion
            if positions[i] != players[0].position and positions[i] != players[1].position:  # If it's held by a player then it's
                # part of the player's state
                object_name = 'soup'
                object_position = positions[i]
                soup_type = 'onion'
                num_onions = 3  # This info isn't in the observation -- presumably it's 3!
                cook_time = 99  # This info isn't in the observation
                soup_state = (soup_type, num_onions, cook_time)
                objects_dict[object_position] = ObjectState(object_name, object_position, soup_state)

        # Soup inside the pot:
        onions_in_pot_pos, onions_in_pot_value = extract_pos_value(state_mask_dict['onions_in_pot'])
        # onions_cook_time_pos, onions_cook_time_value = extract_pos_value(state_mask_dict['onions_cook_time'])
        for i in range(onions_in_pot_pos.__len__()):
            object_name = 'soup'
            object_position = onions_in_pot_pos[i]
            soup_type = 'onion'
            num_onions = onions_in_pot_value[i]
            cook_time = state_mask_dict['onions_cook_time'][object_position]
            soup_state = (soup_type, num_onions, cook_time)
            objects_dict[object_position] = ObjectState(object_name, object_position, soup_state)

        return OvercookedState(players, objects_dict, order_list)

    @staticmethod
    def switch_player(obs):
        """
        Super hacky utility function, that changes the ordering of the layers
        so that the observation becomes in the standard format for the other player
        """
        obs = obs.copy()
        obs = switch_layers(obs, 0, 1)
        obs = switch_layers(obs, 2, 6)
        obs = switch_layers(obs, 3, 7)
        obs = switch_layers(obs, 4, 8)
        obs = switch_layers(obs, 5, 9)
        return obs

def switch_layers(obs, idx0, idx1):
    obs = obs.copy()
    tmp = obs[:,:,idx0].copy()
    obs[:,:,idx0] = obs[:,:,idx1]
    obs[:,:,idx1] = tmp
    return obs


class Direction(object):
    """A class that contains the five actions available in Gridworlds.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    STAY = (0, 0)
    CARDINAL = [NORTH, SOUTH, EAST, WEST]
    INDEX_TO_DIRECTION = CARDINAL + [STAY]
    DIRECTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_DIRECTION) }
    ALL_DIRECTIONS = INDEX_TO_DIRECTION
    OPPOSITE_DIRECTIONS = { NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST }
    ORIENTATION_TO_CHAR = { NORTH: "↑", SOUTH: "↓", EAST: "→", WEST: "←" }

    @staticmethod
    def move_in_direction(point, direction):
        """Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions, except not Direction.STAY or
                   Direction.SELF_LOOP.
        """
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def determine_action_for_change_in_pos(old_pos, new_pos):
        """Determines an action that will enable intended transition"""
        if old_pos == new_pos:
            return Direction.STAY
        new_x, new_y = new_pos
        old_x, old_y = old_pos
        direction = (new_x - old_x, new_y - old_y)
        assert direction in Direction.ALL_DIRECTIONS
        return direction

    @staticmethod
    def get_adjacent_directions(direction):
        """Returns the directions within 90 degrees of the given direction.

        direction: One of the Directions, except not Direction.STAY.
        """
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return [Direction.EAST, Direction.WEST]
        elif direction in [Direction.EAST, Direction.WEST]:
            return [Direction.NORTH, Direction.SOUTH]
        raise ValueError('Invalid direction: %s' % direction)

    @staticmethod
    def get_number_from_direction(direction):
        return Direction.DIRECTION_TO_INDEX[direction]

    @staticmethod
    def get_direction_from_number(number):
        return Direction.INDEX_TO_DIRECTION[number]

class Action(object):
    INTERACT = 'interact'
    INDEX_TO_ACTION = Direction.INDEX_TO_DIRECTION + [INTERACT]
    INDEX_TO_ACTION_INDEX_PAIRS = [v for v in itertools.product(range(len(INDEX_TO_ACTION)), repeat=2)]
    ACTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_ACTION) }
    ALL_ACTIONS = INDEX_TO_ACTION
    MOTION_ACTIONS = Direction.INDEX_TO_DIRECTION