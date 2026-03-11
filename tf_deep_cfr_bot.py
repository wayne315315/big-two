import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Import your existing, unmodified logic
from helper import RANKS, SUITS, evaluate_play, is_valid_beat
from player import Player

# --- NEURAL NETWORK ARCHITECTURE ---
def create_advantage_network(input_size=156, hidden_size=256):
    """
    A Feedforward Neural Network using Keras that predicts the 'regret' 
    (advantage) of taking a specific action in a specific state.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_size,)),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(1) # Outputs a single regret value
    ])
    return model

# --- DEEP CFR PLAYER IMPLEMENTATION ---
class TFDeepCFRBot(Player):
    def __init__(self, name="TF Deep CFR Bot", model_path="tf_advantage_net.weights.h5"):
        super().__init__(name)
        self.model_path = model_path
        
        # Initialize the Keras model
        self.adv_net = create_advantage_network()
        self.load_model()

    def load_model(self):
        """Loads trained weights if available."""
        try:
            self.adv_net.load_weights(self.model_path)
            print(f"[{self.name}] TensorFlow Neural Network loaded successfully.")
        except (FileNotFoundError, OSError):
            print(f"[{self.name}] No pre-trained weights found. Using random initialized weights.")

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
            if is_valid_beat(evaluate_play([card]), game_state.get('table_eval')):
                valid_actions.append([card])
                
        # 2. Pairs
        rank_groups = {}
        for card in self.hand:
            rank_groups[card[0]] = rank_groups.get(card[0], []) + [card]
            
        pairs = [cards[:2] for cards in rank_groups.values() if len(cards) >= 2]
        for pair in pairs:
            if is_valid_beat(evaluate_play(pair), game_state.get('table_eval')):
                valid_actions.append(pair)
                
        # First turn constraint
        if game_state.get('is_first_turn'):
            req_card = game_state['lowest_card']
            valid_actions = [a for a in valid_actions if req_card in a]
            
        return valid_actions

    # --- THE CORE DECISION LOGIC ---
    def get_play(self, game_state):
        """Uses the Keras Neural Network and Regret Matching to select a play."""
        legal_actions = self._get_legal_actions(game_state)
        
        if len(legal_actions) == 1:
            return legal_actions[0] # Forced move
            
        # 1. Encode the current environment state
        hand_arr = self._encode_cards(self.hand)
        table_arr = self._encode_cards(game_state.get('table_cards', []))
        
        # 2. Prepare a single batch of inputs for all legal actions (Much faster in TF!)
        batch_inputs = []
        for action in legal_actions:
            action_arr = self._encode_cards(action)
            # Concatenate state and action into a single 156-length vector
            nn_input = np.concatenate([hand_arr, table_arr, action_arr])
            batch_inputs.append(nn_input)
            
        # Convert list to a 2D numpy array of shape (num_actions, 156)
        batch_inputs = np.array(batch_inputs)
        
        # 3. Ask the Neural Net for the "Regret" of every possible action at once
        predictions = self.adv_net.predict(batch_inputs, verbose=0)
        
        # Flatten the predictions to a 1D list
        advantages = predictions.flatten().tolist()
                
        # 4. Apply Regret Matching Formula
        # We only care about POSITIVE regrets. Negative regret means "don't do this".
        positive_advs = [max(a, 0.0) for a in advantages]
        sum_advs = sum(positive_advs)
        
        if sum_advs > 0:
            probabilities = [a / sum_advs for a in positive_advs]
        else:
            # If the network thinks all moves are bad (or it's untrained), play uniformly random
            probabilities = [1.0 / len(legal_actions)] * len(legal_actions)
            
        # 5. Sample an action based on the Neural Network's probability distribution
        chosen_index = random.choices(range(len(legal_actions)), weights=probabilities, k=1)[0]
        
        return legal_actions[chosen_index]