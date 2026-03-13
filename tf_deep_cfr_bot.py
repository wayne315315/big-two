import random
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Import your existing, unmodified logic
from helper import RANKS, SUITS, evaluate_play, is_valid_beat
from player import Player


# --- NEURAL NETWORK ARCHITECTURE ---
def create_advantage_network(input_size=208, hidden_size=512, num_res_blocks=3):
    """
    Advanced Deep Residual Network for Deep CFR.
    - Wider & Deeper to prevent underfitting.
    - BatchNorm to prevent overfitting.
    """
    inputs = layers.Input(shape=(input_size,))
    
    # 1. Initial Projection Layer (Expands the 208 bits into 512 features)
    x = layers.Dense(hidden_size)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 2. Residual Blocks (Allows deep networks to train without vanishing gradients)
    for _ in range(num_res_blocks):
        residual = x  # Save the input to skip across the block
        
        # First half of the block
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second half of the block
        x = layers.Dense(hidden_size)(x)
        x = layers.BatchNormalization()(x)
        
        # Add the original input back in (The "Skip Connection")
        x = layers.Add()([x, residual]) 
        x = layers.Activation('relu')(x)

    # 3. Value Head (Compresses the complex features down to a single regret value)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Final Output (Linear activation for continuous Regret values)
    outputs = layers.Dense(1)(x)

    return models.Model(inputs=inputs, outputs=outputs)


# --- DEEP CFR PLAYER IMPLEMENTATION ---
class TFDeepCFRBot(Player):
    def __init__(self, name="TF Deep CFR Bot", model_path="tf_advantage_net.weights.h5", model=None, exploration_rate=0.1):
        super().__init__(name)
        self.model_path = model_path
        self.episode_memory = [] # Added for training directly in the main class
        self.exploration_rate = exploration_rate
        
        # Allow sharing an existing model during self-play training
        if model is not None:
            self.adv_net = model
        else:
            self.adv_net = create_advantage_network()
            self.load_model()

    def load_model(self):
        """Loads trained weights if available."""
        try:
            self.adv_net.load_weights(self.model_path)
            #print(f"[{self.name}] TensorFlow Neural Network loaded successfully.")
        except (FileNotFoundError, OSError):
            pass
            #print(f"[{self.name}] No pre-trained weights found. Using random initialized weights.")

    def clear_memory(self):
        """Clears the memory for the next game."""
        self.episode_memory = []

    # --- STATE & ACTION ENCODING ---
    def _card_to_index(self, card):
        """Maps a card tuple ('3', 'Clubs') to a unique integer 0-51."""
        rank, suit = card
        return RANKS.index(rank) * 4 + SUITS.index(suit)

    def _encode_cards(self, cards):
        """Converts a list of cards into a 52-dimensional one-hot numpy array."""
        array = np.zeros(52, dtype=np.float32)
        if cards:
            for card in cards:
                array[self._card_to_index(card)] = 1.0
        return array

    def _get_legal_actions(self, game_state):
        """Generates all legally valid combinations we can play."""
        valid_actions = [[]] if game_state.get('table_eval') else []
        
        # 1. Singles
        for card in self.hand:
            eval_res = evaluate_play([card])
            if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                valid_actions.append([card])
                
        # 2. Pairs (Now properly extracts ALL possible pairs from 3-of-a-kind and 4-of-a-kind)
        rank_groups = {}
        for card in self.hand:
            rank_groups[card[0]] = rank_groups.get(card[0], []) + [card]
            
        for cards in rank_groups.values():
            if len(cards) >= 2:
                # Generate all possible 2-card combinations for this rank
                for pair in itertools.combinations(cards, 2):
                    pair_list = list(pair)
                    eval_res = evaluate_play(pair_list)
                    if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                        valid_actions.append(pair_list)
                        
        # 3. 5-Card Hands (Straights, Full Houses, Quads, Straight Flushes)
        if len(self.hand) >= 5:
            # Generate all possible 5-card combinations from the current hand
            for combo in itertools.combinations(self.hand, 5):
                combo_list = list(combo)
                eval_res = evaluate_play(combo_list)
                # Only append if it is a legally recognized 5-card hand AND beats the table
                if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                    valid_actions.append(combo_list)
                
        # First turn constraint
        if game_state.get('is_first_turn'):
            req_card = game_state['lowest_card']
            valid_actions = [a for a in valid_actions if req_card in a]
            
        return valid_actions

    # --- THE CORE DECISION LOGIC ---
    def get_play(self, game_state):
        """Uses the Keras Neural Network and Regret Matching to select a play."""
        legal_actions = self._get_legal_actions(game_state)
        
        # If there is no choice, pick the only action and return (do not record for training)
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else []
            
        # 1. Encode the current environment state
        hand_arr = self._encode_cards(self.hand)
        table_arr = self._encode_cards(game_state.get('table_cards', []))
        dead_arr = self._encode_cards(game_state.get('dead_cards', [])) # Encode discarded cards
        
        # 2. Prepare a single batch of inputs for all legal actions
        batch_inputs = []
        for action in legal_actions:
            action_arr = self._encode_cards(action)
            # Concatenate state and action into a single 208-length vector
            nn_input = np.concatenate([hand_arr, table_arr, dead_arr, action_arr])
            batch_inputs.append(nn_input)
            
        batch_inputs = np.array(batch_inputs)
        
        # 3. Ask the Neural Net for the "Regret" of every possible action at once
        predictions = self.adv_net.predict(batch_inputs, verbose=0)
        advantages = predictions.flatten().tolist()
                
        # 4. Apply Regret Matching Formula
        positive_advs = [max(a, 0.0) for a in advantages]
        sum_advs = sum(positive_advs)
        
        if sum_advs > 0:
            probabilities = [a / sum_advs for a in positive_advs]
        else:
            probabilities = [1.0 / len(legal_actions)] * len(legal_actions)

        # --- ADDED: EXPLORATION POLICY MIXING ---
        if self.exploration_rate > 0:
            explore_prob = self.exploration_rate / len(legal_actions)
            probabilities = [(p * (1.0 - self.exploration_rate)) + explore_prob for p in probabilities]
            
        # 5. Sample an action based on probabilities
        chosen_index = random.choices(range(len(legal_actions)), weights=probabilities, k=1)[0]
        
        # RECORD STATE-ACTION VECTOR FOR TRAINING
        self.episode_memory.append(batch_inputs[chosen_index])
        
        return legal_actions[chosen_index]