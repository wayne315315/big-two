import random
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from helper import RANKS, SUITS, evaluate_play, is_valid_beat
from player import Player

# ==============================================================================
# NEURAL NETWORK ARCHITECTURES
# ==============================================================================
def create_advantage_network(input_size=208, hidden_size=512, num_res_blocks=3):
    """
    THE TEACHER: Predicts the Raw Expected Value (Q-Value) of taking an action.
    Uses a Deep Residual Network (ResNet) to prevent catastrophic forgetting.
    """
    inputs = layers.Input(shape=(input_size,))
    x = layers.Dense(hidden_size)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Skip connections (Residual Blocks) allow gradients to flow deep into the network
    for _ in range(num_res_blocks):
        residual = x
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])  # Add original input back (Skip connection)
        x = layers.Activation('relu')(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(1)(x) 
    outputs = layers.Activation('linear')(x) # Linear for Q-Values

    return models.Model(inputs=inputs, outputs=outputs)

def create_policy_network(input_size=208, hidden_size=512, num_res_blocks=3):
    """
    THE STUDENT: Watches the Teacher and averages out all strategies to form the Nash Equilibrium.
    Must have the exact same capacity as the Advantage network to memorize the strategy perfectly.
    """
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
    
    x = layers.Dense(1)(x)
    outputs = layers.Activation('sigmoid')(x) # Sigmoid for Probabilities (0.0 to 1.0)

    return models.Model(inputs=inputs, outputs=outputs)

# ==============================================================================
# TRUE DEEP CFR PLAYER (CLIENT ENABLED)
# ==============================================================================
class TFDeepCFRBot(Player):
    def __init__(self, name="TF Deep CFR Bot", adv_model_path="tf_advantage_model.keras", 
                 policy_model_path="tf_policy_model.keras", is_training=True, exploration_rate=0.1, pipe=None):
        super().__init__(name)
        self.is_training = is_training
        self.exploration_rate = exploration_rate if is_training else 0.0
        self.pipe = pipe  # IPC Pipe for sending math to the GPU Server
        
        self.episode_memory = [] # Tracks moves to calculate Regret later
        self.policy_memory = []  # Tracks probabilities to teach the Policy Network
        
        # If no pipe is provided (e.g., standalone app), load the heavy networks into local RAM
        if self.pipe is None:
            # OPTIMIZATION: Load full models compiled=False for lightning-fast local inference
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
        rank, suit = card
        return RANKS.index(rank) * 4 + SUITS.index(suit)

    def _encode_cards(self, cards):
        # FAST CPU EXECUTION: Let the CPU use native float32
        array = np.zeros(52, dtype=np.float32)
        if cards:
            for card in cards:
                array[self._card_to_index(card)] = 1.0
        return array

    def _get_legal_actions(self, game_state):
        """Generates all mathematically legal moves according to Big Two rules."""
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
        
        # Auto-play if there's no choice
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else []
            
        # Encode the game environment into a tensor
        hand_arr = self._encode_cards(self.hand)
        table_arr = self._encode_cards(game_state.get('table_cards', []))
        dead_arr = self._encode_cards(game_state.get('dead_cards', []))
        
        batch_inputs = []
        for action in legal_actions:
            action_arr = self._encode_cards(action)
            batch_inputs.append(np.concatenate([hand_arr, table_arr, dead_arr, action_arr]))
        batch_inputs = np.array(batch_inputs)
        
        if self.is_training:
            # ==================================================================
            # PHASE 1: EXPLORATION (Teacher Network calculates Q-Values)
            # ==================================================================
            
            # Send tensor over the pipe to the GPU (or evaluate locally)
            if self.pipe:
                self.pipe.send((False, batch_inputs)) # False = "Use Advantage Network"
                q_values = self.pipe.recv().flatten().tolist()
            else:
                q_values = self.adv_net(batch_inputs, training=False).numpy().flatten().tolist()
            
            # CFR MATH: Calculate Baseline Expected Value V(s)
            baseline_value = sum(q_values) / len(q_values)
            
            # CFR MATH: Regret is the Q-value MINUS the Baseline
            advantages = [q - baseline_value for q in q_values]
                    
            # REGRET MATCHING: Convert positive advantages into probabilities
            positive_advs = [max(a, 0.0) for a in advantages]
            sum_advs = sum(positive_advs)
            
            if sum_advs > 0: probabilities = [a / sum_advs for a in positive_advs]
            else: probabilities = [1.0 / len(legal_actions)] * len(legal_actions) # Fallback

            # EXPLORATION: Inject random noise so the bot tries new strategies
            if self.exploration_rate > 0:
                explore_prob = self.exploration_rate / len(legal_actions)
                final_probs = [(p * (1.0 - self.exploration_rate)) + explore_prob for p in probabilities]
            else:
                final_probs = probabilities
                
            # Pick a move based on the generated probability distribution
            chosen_index = random.choices(range(len(legal_actions)), weights=final_probs, k=1)[0]
            
            # Store the true state value so the worker can calculate the Final Regret later
            self.episode_memory.append({
                'inputs': batch_inputs, 'chosen_index': chosen_index, 'baseline_value': baseline_value
            })
            
            # Store the probability so the Policy Network can learn it
            for i in range(len(legal_actions)):
                self.policy_memory.append((batch_inputs[i], probabilities[i]))
                
            return legal_actions[chosen_index]
            
        else:
            # ==================================================================
            # PHASE 2: EXPLOITATION (Student Network plays the Nash Equilibrium)
            # ==================================================================
            if self.pipe:
                self.pipe.send((True, batch_inputs)) # True = "Use Policy Network"
                policy_probs = self.pipe.recv().flatten().tolist()
            else:
                policy_probs = self.policy_net(batch_inputs, training=False).numpy().flatten().tolist()
                
            # Act greedily (100% confidence) on the best move 
            best_action_idx = np.argmax(policy_probs)
            return legal_actions[best_action_idx]
