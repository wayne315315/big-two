import random
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from helper import RANKS, SUITS, evaluate_play, is_valid_beat
from player import Player

# --- NEURAL NETWORK ARCHITECTURES ---
def create_advantage_network(input_size=208, hidden_size=512, num_res_blocks=3):
    """Predicts the Regret of taking an action."""
    inputs = layers.Input(shape=(input_size,))
    x = layers.Dense(hidden_size)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for _ in range(num_res_blocks):
        residual = x
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual]) 
        x = layers.Activation('relu')(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1)(x) # Linear output for continuous regret

    return models.Model(inputs=inputs, outputs=outputs)

def create_policy_network(input_size=208, hidden_size=512, num_res_blocks=3):
    """Predicts the average optimal probability of taking an action (Nash Equilibrium)."""
    inputs = layers.Input(shape=(input_size,))
    x = layers.Dense(hidden_size)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for _ in range(num_res_blocks):
        residual = x
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual]) 
        x = layers.Activation('relu')(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x) # Sigmoid output for probabilities

    return models.Model(inputs=inputs, outputs=outputs)

# --- TRUE DEEP CFR PLAYER ---
class TFDeepCFRBot(Player):
    def __init__(self, name="TF Deep CFR Bot", adv_model_path="tf_advantage_net.weights.h5", 
                 policy_model_path="tf_policy_net.weights.h5", is_training=True, exploration_rate=0.1):
        super().__init__(name)
        self.adv_model_path = adv_model_path
        self.policy_model_path = policy_model_path
        self.is_training = is_training
        self.exploration_rate = exploration_rate if is_training else 0.0
        
        self.episode_memory = [] # For Regret
        self.policy_memory = []  # For Average Strategy
        
        self.adv_net = create_advantage_network()
        self.policy_net = create_policy_network()
        self.load_models()

    def load_models(self):
        try: self.adv_net.load_weights(self.adv_model_path)
        except: pass
        try: self.policy_net.load_weights(self.policy_model_path)
        except: pass

    def clear_memory(self):
        self.episode_memory = []
        self.policy_memory = []

    def _card_to_index(self, card):
        rank, suit = card
        return RANKS.index(rank) * 4 + SUITS.index(suit)

    def _encode_cards(self, cards):
        array = np.zeros(52, dtype=np.float32)
        if cards:
            for card in cards:
                array[self._card_to_index(card)] = 1.0
        return array

    def _get_legal_actions(self, game_state):
        valid_actions = [[]] if game_state.get('table_eval') else []
        
        for card in self.hand:
            eval_res = evaluate_play([card])
            if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                valid_actions.append([card])
                
        rank_groups = {}
        for card in self.hand:
            rank_groups[card[0]] = rank_groups.get(card[0], []) + [card]
            
        for cards in rank_groups.values():
            if len(cards) >= 2:
                for pair in itertools.combinations(cards, 2):
                    pair_list = list(pair)
                    eval_res = evaluate_play(pair_list)
                    if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                        valid_actions.append(pair_list)
                        
        if len(self.hand) >= 5:
            for combo in itertools.combinations(self.hand, 5):
                combo_list = list(combo)
                eval_res = evaluate_play(combo_list)
                if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                    valid_actions.append(combo_list)
                
        if game_state.get('is_first_turn'):
            req_card = game_state['lowest_card']
            valid_actions = [a for a in valid_actions if req_card in a]
            
        return valid_actions

    def get_play(self, game_state):
        legal_actions = self._get_legal_actions(game_state)
        
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else []
            
        hand_arr = self._encode_cards(self.hand)
        table_arr = self._encode_cards(game_state.get('table_cards', []))
        dead_arr = self._encode_cards(game_state.get('dead_cards', []))
        
        batch_inputs = []
        for action in legal_actions:
            action_arr = self._encode_cards(action)
            batch_inputs.append(np.concatenate([hand_arr, table_arr, dead_arr, action_arr]))
        batch_inputs = np.array(batch_inputs)
        
        if self.is_training:
            # Phase 1: Exploration and Regret Calculation
            predictions = self.adv_net.predict(batch_inputs, verbose=0)
            advantages = predictions.flatten().tolist()
                    
            positive_advs = [max(a, 0.0) for a in advantages]
            sum_advs = sum(positive_advs)
            
            if sum_advs > 0:
                probabilities = [a / sum_advs for a in positive_advs]
            else:
                probabilities = [1.0 / len(legal_actions)] * len(legal_actions)

            if self.exploration_rate > 0:
                explore_prob = self.exploration_rate / len(legal_actions)
                final_probs = [(p * (1.0 - self.exploration_rate)) + explore_prob for p in probabilities]
            else:
                final_probs = probabilities
                
            chosen_index = random.choices(range(len(legal_actions)), weights=final_probs, k=1)[0]
            
            # Baseline V(s) calculation
            state_value = sum([p * adv for p, adv in zip(probabilities, advantages)])
            
            self.episode_memory.append({
                'inputs': batch_inputs,
                'chosen_index': chosen_index,
                'baseline_value': state_value
            })
            
            # Target probabilities for Average Policy Network
            for i in range(len(legal_actions)):
                self.policy_memory.append((batch_inputs[i], probabilities[i]))
            
            return legal_actions[chosen_index]
        else:
            # Phase 2: Pure Exploitation using the Nash Equilibrium
            predictions = self.policy_net.predict(batch_inputs, verbose=0)
            policy_probs = predictions.flatten().tolist()
            
            best_action_idx = np.argmax(policy_probs)
            return legal_actions[best_action_idx]
