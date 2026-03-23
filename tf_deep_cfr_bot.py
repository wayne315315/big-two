import random
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision

# Corrected Import: Removed RANK_MAP and SUIT_MAP
from helper import RANKS, SUITS, evaluate_play, is_valid_beat
from player import Player

mixed_precision.set_global_policy('mixed_float16')

# ==============================================================================
# LOCAL HASH MAPS (O(1) Lookups without touching helper.py)
# ==============================================================================
RANK_MAP = {r: i for i, r in enumerate(RANKS)}
SUIT_MAP = {s: i for i, s in enumerate(SUITS)}

# ==============================================================================
# TEMPORAL 1D RESNET ARCHITECTURE
# ==============================================================================
def conv1d_residual_block(x, filters, kernel_size=3, dropout_rate=0.1):
    shortcut = x
    
    # 1D Convolution slides down the 37 sequence steps to learn temporal flow
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def create_temporal_resnet(input_shape=(37, 55), num_blocks=5, filters=256, is_policy=False):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    for _ in range(num_blocks):
        x = conv1d_residual_block(x, filters)
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # MIXED PRECISION: Final output MUST be float32
    x = layers.Dense(1)(x)
    if is_policy:
        outputs = layers.Activation('sigmoid', dtype='float32')(x)
    else:
        outputs = layers.Activation('linear', dtype='float32')(x) 

    return models.Model(inputs=inputs, outputs=outputs)

def create_advantage_network():
    return create_temporal_resnet(is_policy=False)

def create_policy_network():
    return create_temporal_resnet(is_policy=True)

# ==============================================================================
# TRUE DEEP CFR PLAYER (ORDERED TEMPORAL MATRIX)
# ==============================================================================
class TFDeepCFRBot(Player):
    def __init__(self, name="TF Deep CFR Bot", adv_model_path="tf_advantage_model.keras", 
                 policy_model_path="tf_policy_model.keras", is_training=True, exploration_rate=0.1, pipe=None):
        super().__init__(name)
        self.is_training = is_training
        self.exploration_rate = exploration_rate if is_training else 0.0
        self.pipe = pipe  
        
        self.episode_memory = [] 
        self.policy_memory = []  
        
        if self.pipe is None:
            try:
                self.adv_net = tf.keras.models.load_model(adv_model_path, compile=False)
            except:
                self.adv_net = create_advantage_network()
            try:
                self.policy_net = tf.keras.models.load_model(policy_model_path, compile=False)
            except:
                self.policy_net = create_policy_network()

    def clear_memory(self):
        self.episode_memory = []
        self.policy_memory = []

    def _card_to_index(self, card):
        return RANK_MAP[card[0]] * 4 + SUIT_MAP[card[1]]

    def _encode_base_sequence(self, game_state):
        MAX_SEQ_LEN = 37 
        FEATURE_DIM = 55
        seq = np.zeros((MAX_SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        
        my_idx = game_state.get('my_idx', 0)
        current_my_size = game_state.get('my_hand_size', 13) 
        current_opp_size = game_state.get('opp_hand_size', 13) 
        
        my_s_norm = current_my_size / 13.0
        opp_s_norm = current_opp_size / 13.0

        for c in self.hand: seq[0, self._card_to_index(c)] = 1.0
        seq[0, 52], seq[0, 53], seq[0, 54] = 1.0, my_s_norm, opp_s_norm
        
        for c in game_state.get('table_cards', []): seq[1, self._card_to_index(c)] = 1.0
        seq[1, 52], seq[1, 53], seq[1, 54] = 0.0, my_s_norm, opp_s_norm
        
        for c in game_state.get('dead_cards', []): seq[3, self._card_to_index(c)] = 1.0
        seq[3, 52], seq[3, 53], seq[3, 54] = 0.0, my_s_norm, opp_s_norm
        
        history = game_state.get('history', [])
        hist_len = min(len(history), MAX_SEQ_LEN - 4)
        
        hist_my_size = current_my_size
        hist_opp_size = current_opp_size
        
        # Chronological history injection for Conv1D to read as a timeline
        for i in range(hist_len):
            row_idx = 4 + i
            hist_item = history[-(i+1)] 
            hist_player_idx, hist_cards = hist_item[0], hist_item[1]
            
            for c in hist_cards:
                seq[row_idx, self._card_to_index(c)] = 1.0
                
            seq[row_idx, 52] = 1.0 if hist_player_idx == my_idx else -1.0
            seq[row_idx, 53] = hist_my_size / 13.0
            seq[row_idx, 54] = hist_opp_size / 13.0
            
            if hist_player_idx == my_idx:
                hist_my_size += len(hist_cards)
            else:
                hist_opp_size += len(hist_cards)
            
        return seq

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
            
        base_sequence = self._encode_base_sequence(game_state)
        batch_inputs = np.empty((len(legal_actions), 37, 55), dtype=np.float32)
        
        my_s_norm = game_state.get('my_hand_size', 13) / 13.0
        opp_s_norm = game_state.get('opp_hand_size', 13) / 13.0

        for i, action in enumerate(legal_actions):
            np.copyto(batch_inputs[i], base_sequence) 
            for c in action:
                batch_inputs[i, 2, self._card_to_index(c)] = 1.0
            batch_inputs[i, 2, 52] = 1.0
            batch_inputs[i, 2, 53] = my_s_norm
            batch_inputs[i, 2, 54] = opp_s_norm
        
        if self.is_training:
            if self.pipe:
                self.pipe.send((False, batch_inputs.astype(np.float16)))
                advantages = self.pipe.recv().flatten().tolist()
            else:
                advantages = self.adv_net(batch_inputs, training=False).numpy().flatten().tolist()
            
            baseline_value = sum(advantages) / len(advantages)
            regrets = [adv - baseline_value for adv in advantages]
                    
            positive_advs = [max(a, 0.0) for a in regrets]
            sum_advs = sum(positive_advs)
            
            if sum_advs > 0: probabilities = [a / sum_advs for a in positive_advs]
            else: probabilities = [1.0 / len(legal_actions)] * len(legal_actions) 

            if self.exploration_rate > 0:
                explore_prob = self.exploration_rate / len(legal_actions)
                final_probs = [(p * (1.0 - self.exploration_rate)) + explore_prob for p in probabilities]
            else:
                final_probs = probabilities
                
            chosen_index = random.choices(range(len(legal_actions)), weights=final_probs, k=1)[0]
            
            self.episode_memory.append({
                'inputs': batch_inputs, 'chosen_index': chosen_index, 'baseline_value': baseline_value
            })
            for i in range(len(legal_actions)):
                self.policy_memory.append((batch_inputs[i], probabilities[i]))
            return legal_actions[chosen_index]
            
        else:
            if self.pipe:
                self.pipe.send((True, batch_inputs.astype(np.float16)))
                policy_probs = self.pipe.recv().flatten().tolist()
            else:
                policy_probs = self.policy_net(batch_inputs, training=False).numpy().flatten().tolist()
                
            best_action_idx = np.argmax(policy_probs)
            return legal_actions[best_action_idx]
