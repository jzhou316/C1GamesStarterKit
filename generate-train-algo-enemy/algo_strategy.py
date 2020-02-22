import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import torch
from model import QModel
import numpy as np
import os

global current_dir 
current_dir = os.path.dirname(__file__)
current_dir = '../../saved_models'

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER, BITS, CORES
        global NEED_DUMP
        NEED_DUMP = False
        self.need_dump = NEED_DUMP
        FILTER = config["unitInformation"][0]["shorthand"]
        ENCRYPTOR = config["unitInformation"][1]["shorthand"]
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        PING = config["unitInformation"][3]["shorthand"]
        EMP = config["unitInformation"][4]["shorthand"]
        SCRAMBLER = config["unitInformation"][5]["shorthand"]
        BITS = 1
        CORES = 0
        # This is a good place to do initial setup
        self.scored_on_locations = []
        self.to_dump = []
        s_dim = 3136  # 28 * 28 * 3
        a_att_dim = 840  # 28 * 3 * 10
        a_def_dim = 1568  # 28 * 14 * 3
        # total 4368, #params 4368 * 500 = 2,184,000
        hid_dim = 500
        self.model = QModel(s_dim, a_att_dim, a_def_dim, hid_dim)
        gamelib.debug_write('---------------------------------------------------loading model from %s/model.pt'%current_dir)
        #self.model.load(os.path.join(current_dir, 'model.pt'))

    
        

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.
        
        state = self.game_state_to_tensor(game_state)

        if game_state.turn_number == 0:
            p = random.random()
            p = 0.5
            if p < 0.2:
                #######
                # Start_algo
                game_state.attemp_spawn(FILTER, [0,13], 1)
                 # Place destructors that attack enemy units
                destructor_locations = [[0, 13], [27, 13], [8, 11], [19, 11], [13, 11], [14, 11]]
                # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
                game_state.attempt_spawn(DESTRUCTOR, destructor_locations)
        
                # Place filters in front of destructors to soak up damage for them
                filter_locations = [[8, 12], [19, 12]]
                game_state.attempt_spawn(FILTER, filter_locations)
                # upgrade filters so they soak more damage
                game_state.attempt_upgrade(filter_locations)
                ####End start_algo 
            elif p<0.4:
                gamelib.debug_write('BOSE 1')
                ######
                # Bose_1 
                destructor_locations = [[0, 14], [1, 14], [3, 14], [24, 14], [26, 14], [27, 14]]
                # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
                game_state.attempt_spawn(DESTRUCTOR, destructor_locations)
        
                # Place filters in front of destructors to soak up damage for them
                filter_locations = [[2, 12], [4, 12],[23, 12], [25, 12]]
                game_state.attempt_spawn(FILTER, filter_locations)
                # upgrade filters so they soak more damage
                game_state.attempt_upgrade([[10,22],[12,22],[13,20]])
             
            elif p<0.6:
                gamelib.debug_write('BOSE 2')
                ######
                # Bose_2 
                destructor_locations = [[4, 16], [13, 21], [23, 16]]
                # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
                game_state.attempt_spawn(DESTRUCTOR, destructor_locations)
        
                # Place filters in front of destructors to soak up damage for them
                filter_locations = [[2, 14], [3, 15],[5, 17], [6, 18],[7, 19], [8, 20],[9, 21], [10, 22],[11, 23], [12, 22],[13, 20], [14, 20],[15, 20], [16, 20],[17, 20], [18, 20],[19, 20],[20, 19], [19, 20],[18, 21], [24, 15],[25, 14]]
                game_state.attempt_spawn(FILTER, filter_locations)
                ######End of bose 2
             
            elif p<0.8:
                gamelib.debug_write('BOSE 3')
                ######
                # Bose_3 
                destructor_locations = [[4, 16], [13, 21], [23, 16]]
                # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
                game_state.attempt_spawn(DESTRUCTOR, destructor_locations)
        
                # Place filters in front of destructors to soak up damage for them
                filter_locations = [[2, 14], [3, 15],[5, 17], [6, 18],[7, 19], [8, 20],[9, 21], [10, 22],[11, 23], [12, 22],[13, 20], [14, 20],[15, 20], [16, 20],[17, 20], [18, 20],[19, 20],[20, 19], [19, 20],[18, 21], [24, 15],[25, 14]]
                game_state.attempt_spawn(FILTER, filter_locations)
             
            elif p<0.8:
                ######
                # Bose_4
                destructor_locations = [[4, 16], [13, 21], [23, 16]]
                # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
                game_state.attempt_spawn(DESTRUCTOR, destructor_locations)
                # Place filters in front of destructors to soak up damage for them
                filter_locations = [[2, 14], [3, 15],[5, 17], [6, 18],[7, 19], [8, 20],[9, 21], [10, 22],[11, 23], [12, 22],[13, 20], [14, 20],[15, 20], [16, 20],[17, 20], [18, 20],[19, 20],[20, 19], [19, 20],[18, 21], [24, 15],[25, 14]]
                game_state.attempt_spawn(FILTER, filter_locations)
                #####End of bose 4 
              
              
        
        p = random.random()
        if p < 0.8:
            gamelib.debug_write('Using BASELINE Policy')
            self.starter_strategy(game_state)
        else:
            gamelib.debug_write('Using RL Policy')
            _, _, scores_attack, scores_defense = self.model.infer_act(state.float())
            self.tensor_to_attack_defense(game_state, scores_attack, scores_defense)

        build_stack = game_state._build_stack
        deploy_stack = game_state._deploy_stack
        action_defense = self.build_to_defense(build_stack)
        action_attack = self.build_to_attack(deploy_stack)

            #for unit_type, x, y in game_state._build_stack:
            #    costs = game_state.type_cost(unit_type)
            #    game_state.__set_resource(0, 0 + costs[0])
            #    game_state.__set_resource(1, 0 + costs[1])

            #game_state._build_stack = []
            #game_state._deploy_stack = []


        my_health = game_state.my_health
        enemy_health = game_state.enemy_health
        self.to_dump.append((state, action_attack, action_defense, my_health, enemy_health))
        game_state.submit_turn()


        #return a_att.reshape(28, 4, 10), a_def.reshape(28, 14, 3), a_grads[0].reshape(28, 4, 10), a_grads[1].reshape(28, 14, 3)
    def tensor_to_attack_defense(self, game_state, scores_attack, scores_defense):
        attack_id2name = {0: PING,
                   1: EMP,
                   2: SCRAMBLER}
        defense_id2name = {0: FILTER,
                   1: ENCRYPTOR,
                   2: DESTRUCTOR}
        scores_attack = scores_attack.view(-1)
        scores_defense = scores_defense.view(-1)

        sorted_attack, attack_ids = torch.sort(scores_attack, 0, descending=True)
        sorted_defense, defense_ids = torch.sort(scores_defense, 0, descending=True)

        attack_ptr, defense_ptr = 0, 0

        def attempt_defense(defense_id):
            defense_id = defense_id.item()
            x = int( math.floor(defense_id / (4*14)))
            defense_id = defense_id - x * (4*14)
            y = int(math.floor(defense_id / (4)))
            defense_id = defense_id - y * 4
            z = defense_id
            if z == 3:
                game_state.attempt_upgrade([(x, y)])
                return
            game_state.attempt_spawn(defense_id2name[z], (x, y))
            gamelib.debug_write('TENSOR_TO_DEFENSE x: %s, y: %s, id: %s...'%(x, y, z))

        def attempt_attack(attack_id):
            attack_id = attack_id.item()
            x = int( math.floor(attack_id / (3*10)))
            attack_id = attack_id - x * (3*10)
            y = int(math.floor(attack_id / (10)))
            attack_id = attack_id - y * 10
            z = attack_id
            if x < 14:
                game_state.attempt_spawn(attack_id2name[y], [x,13-x], z+1)
            else:
                game_state.attempt_spawn(attack_id2name[y], [x,x-14], z+1)
            gamelib.debug_write('TENSOR_TO_ATTACK x: %s, id: %s, num: %s...'%(x, y, z+1))

        while attack_ptr < scores_attack.size(0) or defense_ptr < scores_defense.size(0):
            if attack_ptr == scores_attack.size(0):
                if sorted_defense[defense_ptr].item() > 0:
                    attempt_defense(defense_ids[defense_ptr])
                    defense_ptr += 1
                else:
                    break
            elif defense_ptr == scores_defense.size(0):
                if sorted_attack[attack_ptr].item() > 0:
                    attempt_attack(attack_ids[attack_ptr])
                    attack_ptr += 1
                else:
                    break
            else:
                if sorted_attack[attack_ptr] > sorted_defense[defense_ptr]:
                    if sorted_attack[attack_ptr].item() < 0:
                        break
                    attempt_attack(attack_ids[attack_ptr])
                    attack_ptr += 1
                else:
                    if sorted_defense[defense_ptr].item() < 0:
                        break
                    attempt_defense(defense_ids[defense_ptr])
                    defense_ptr += 1

        return


    def build_to_attack(self, build_stack):
        name2id = {PING:0,
           EMP:1,
           SCRAMBLER:2}
        array = torch.zeros((28, 3)).long()
        action_attack = torch.zeros(28, 3, 10).long()
        for obj in build_stack:
            unit_type = obj[0]
            x = obj[1]
            array[x, name2id[unit_type]] += 1

        for obj in build_stack:
            unit_type = obj[0]
            x = obj[1]
            num = array[x, name2id[unit_type]]
            action_attack[x, name2id[unit_type], min(num.item()-1, 9)] = 1
        return action_attack

    def build_to_defense(self, deploy_stack):
        name2id = {FILTER: 0,
                   ENCRYPTOR : 1,
                   DESTRUCTOR : 2}
        action_defense = torch.zeros(28, 14, 4).long()
        for obj in deploy_stack:
            unit_type = obj[0]
            x = obj[1]
            y = obj[2]
            if unit_type == 'UP':
                action_defense[x, y, 3] = 1
                continue
            action_defense[x, y, name2id[unit_type]] = 1
        return action_defense

    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """

    def starter_strategy(self, game_state):
        """
        For defense we will use a spread out layout and some Scramblers early on.
        We will place destructors near locations the opponent managed to score on.
        For offense we will use long range EMPs if they place stationary units near the enemy's front.
        If there are no stationary units to attack in the front, we will send Pings to try and score quickly.
        """
        # First, place basic defenses
        self.build_defences(game_state)
        # Now build reactive defenses based on where the enemy scored
        self.build_reactive_defense(game_state)

        # If the turn is less than 5, stall with Scramblers and wait to see enemy's base
        if game_state.turn_number < 5:
            self.stall_with_scramblers(game_state)
        else:
            # Now let's analyze the enemy base to see where their defenses are concentrated.
            # If they have many units in the front we can build a line for our EMPs to attack them at long range.
            if self.detect_enemy_unit(game_state, unit_type=None, valid_x=None, valid_y=[14, 15]) > 10:
                self.emp_line_strategy(game_state)
            else:
                # They don't have many units in the front so lets figure out their least defended area and send Pings there.

                # Only spawn Ping's every other turn
                # Sending more at once is better since attacks can only hit a single ping at a time
                if game_state.turn_number % 2 == 1:
                    # To simplify we will just check sending them from back left and right
                    ping_spawn_location_options = [[13, 0], [14, 0]]
                    best_location = self.least_damage_spawn_location(game_state, ping_spawn_location_options)
                    game_state.attempt_spawn(PING, best_location, 1000)

                # Lastly, if we have spare cores, let's build some Encryptors to boost our Pings' health.
                encryptor_locations = [[13, 2], [14, 2], [13, 3], [14, 3]]
                game_state.attempt_spawn(ENCRYPTOR, encryptor_locations)

    def build_defences(self, game_state):
        """
        Build basic defenses using hardcoded locations.
        Remember to defend corners and avoid placing units in the front where enemy EMPs can attack them.
        """
        # Useful tool for setting up your base locations: https://www.kevinbai.design/terminal-map-maker
        # More community tools available at: https://terminal.c1games.com/rules#Download

        # Place destructors that attack enemy units
        destructor_locations = [[0, 13], [27, 13], [8, 11], [19, 11], [13, 11], [14, 11]]
        # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
        game_state.attempt_spawn(DESTRUCTOR, destructor_locations)
        
        # Place filters in front of destructors to soak up damage for them
        filter_locations = [[8, 12], [19, 12]]
        game_state.attempt_spawn(FILTER, filter_locations)
        # upgrade filters so they soak more damage
        if not NEED_DUMP:
            game_state.attempt_upgrade(filter_locations)

    def build_reactive_defense(self, game_state):
        """
        This function builds reactive defenses based on where the enemy scored on us from.
        We can track where the opponent scored by looking at events in action frames 
        as shown in the on_action_frame function
        """
        for location in self.scored_on_locations:
            # Build destructor one space above so that it doesn't block our own edge spawn locations
            build_location = [location[0], location[1]+1]
            game_state.attempt_spawn(DESTRUCTOR, build_location)

    def stall_with_scramblers(self, game_state):
        """
        Send out Scramblers at random locations to defend our base from enemy moving units.
        """
        # We can spawn moving units on our edges so a list of all our edge locations
        friendly_edges = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        
        # Remove locations that are blocked by our own firewalls 
        # since we can't deploy units there.
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        
        # While we have remaining bits to spend lets send out scramblers randomly.
        deployed = 0
        while game_state.get_resource(BITS) >= game_state.type_cost(SCRAMBLER)[BITS] and len(deploy_locations) > 0:
            # Choose a random deploy location.
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]
            
            game_state.attempt_spawn(SCRAMBLER, deploy_location)
            deployed += 1
            if NEED_DUMP and deployed == 10:
                break
            """
            We don't have to remove the location since multiple information 
            units can occupy the same space.
            """

    def emp_line_strategy(self, game_state):
        """
        Build a line of the cheapest stationary unit so our EMP's can attack from long range.
        """
        # First let's figure out the cheapest unit
        # We could just check the game rules, but this demonstrates how to use the GameUnit class
        stationary_units = [FILTER, DESTRUCTOR, ENCRYPTOR]
        cheapest_unit = FILTER
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.BITS] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.BITS]:
                cheapest_unit = unit

        # Now let's build out a line of stationary units. This will prevent our EMPs from running into the enemy base.
        # Instead they will stay at the perfect distance to attack the front two rows of the enemy base.
        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])

        # Now spawn EMPs next to the line
        # By asking attempt_spawn to spawn 1000 units, it will essentially spawn as many as we have resources for
        game_state.attempt_spawn(EMP, [24, 10], 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        This function will help us guess which location is the safest to spawn moving units from.
        It gets the path the unit will take then checks locations on that path to 
        estimate the path's damage risk.
        """
        damages = []
        # Get the damage estimate each path will take
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                # Get number of enemy destructors that can attack the final location and multiply by destructor damage
                damage += len(game_state.get_attackers(path_location, 0)) * gamelib.GameUnit(DESTRUCTOR, game_state.config).damage_i
            damages.append(damage)
        
        # Now just return the location that takes the least damage
        return location_options[damages.index(min(damages))]

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x = None, valid_y = None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1 and (unit_type is None or unit.unit_type == unit_type) and (valid_x is None or location[0] in valid_x) and (valid_y is None or location[1] in valid_y):
                        total_units += 1
        return total_units
        
    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at: https://docs.c1games.com/json-docs.html
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))
    
    def game_state_to_tensor(self, game_state):
        state = torch.zeros(28, 28, 4).long()
        name2id = {FILTER: 0, 
                   ENCRYPTOR: 1,
                   DESTRUCTOR: 2}
        for i in range(28):
            for j in range(28):
                import sys
                if game_state.game_map[i, j] is not None:
                    units = game_state.game_map[i, j]
                    if len(units) > 0:
                        assert len(units) == 1, len(units)
                        unit = units[-1]
                        name = unit.unit_type
                        assert name in name2id, name
                        state[i, j, name2id[name]] = 1
                        if unit.upgraded:
                            state[i, j, 3] = 1
        return state


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
