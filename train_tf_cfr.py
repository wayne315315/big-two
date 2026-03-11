import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
import time

# Import your unmodified rules and base player
from helper import RANKS, SUITS, get_card_value, evaluate_play, is_valid_beat
from tf_deep_cfr_bot import TFDeepCFRBot, create_advantage_network

class TrainingCFRBot(TFDeepCFRBot):
    """
    An extension of the TFDeepCFRBot that records its state-action history 
    during a game so we can train the neural network afterward.
    """
    def __init__(self, name, shared_model):
        super().__init__(name)
        self.adv_net = shared_model # Share the same brain between both players
        self.episode_memory = []    # Stores the (state_action_vector) for the current game

    def get_play(self, game_state):
        legal_actions = self._get_legal_actions(game_state)
        
        # If forced to pass or only one move is available, just play it without recording
        # (There is nothing for the neural network to "learn" if there is no choice)
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else []
            
        hand_arr = self._encode_cards(self.hand)
        table_arr = self._encode_cards(game_state.get('table_cards', []))
        
        batch_inputs = []
        for action in legal_actions:
            action_arr = self._encode_cards(action)
            batch_inputs.append(np.concatenate([hand_arr, table_arr, action_arr]))
            
        batch_inputs = np.array(batch_inputs)
        
        # Predict advantages for all legal actions
        predictions = self.adv_net.predict(batch_inputs, verbose=0).flatten()
        positive_advs = [max(a, 0.0) for a in predictions]
        sum_advs = sum(positive_advs)
        
        if sum_advs > 0:
            probabilities = [a / sum_advs for a in positive_advs]
        else:
            probabilities = [1.0 / len(legal_actions)] * len(legal_actions)
            
        # Sample an action based on probabilities
        #chosen_idx = np.random.choice(len(legal_actions), p=probabilities)
        chosen_idx = random.choices(range(len(legal_actions)), weights=probabilities, k=1)[0]
        chosen_action = legal_actions[chosen_idx]
        
        # --- TRAINING ADDITION ---
        # Record the exact state-action vector that was chosen into memory
        self.episode_memory.append(batch_inputs[chosen_idx])
        
        return chosen_action

    def clear_memory(self):
        """Clears the memory for the next game."""
        self.episode_memory = []


def train_self_play(episodes=1000, batch_size=64):
    """
    The main Monte Carlo Counterfactual Regret self-play training loop.
    """
    print("="*50)
    print("INITIALIZING DEEP CFR SELF-PLAY TRAINING (TENSORFLOW)")
    print("="*50)

    # 1. Create the shared Neural Network and the optimizer
    shared_model = create_advantage_network()
    optimizer = optimizers.Adam(learning_rate=0.001)
    shared_model.compile(optimizer=optimizer, loss='mse')

    # 2. Setup the training bots
    bot1 = TrainingCFRBot("Player 1", shared_model)
    bot2 = TrainingCFRBot("Player 2", shared_model)
    players = [bot1, bot2]

    # --- METRICS TRACKING ---
    metrics = {
        'p1_wins': 0,
        'p2_wins': 0,
        'game_lengths': [],
        'losses': []
    }

    start_time = time.time()

    # 3. Main Training Loop
    for episode in range(1, episodes + 1):
        # Reset deck and hands
        deck = [(rank, suit) for rank in RANKS for suit in SUITS]
        random.shuffle(deck)
        piles = [deck[i * 13 : (i + 1) * 13] for i in range(4)]
        
        bot1.hand = []
        bot2.hand = []
        bot1.receive_cards(piles[0])
        bot2.receive_cards(piles[1])
        
        bot1.clear_memory()
        bot2.clear_memory()

        # Determine starter
        if get_card_value(bot1.hand[0]) < get_card_value(bot2.hand[0]):
            current_idx = 0
            lowest_card = bot1.hand[0]
        else:
            current_idx = 1
            lowest_card = bot2.hand[0]

        game_state = {
            'table_eval': None,
            'table_cards': [],
            'is_first_turn': True,
            'lowest_card': lowest_card
        }
        
        last_player_idx = None
        turns_played = 0

        # --- PLAY A SINGLE GAME ---
        while True:
            current_player = players[current_idx]
            
            if last_player_idx == current_idx:
                game_state['table_eval'] = None
                game_state['table_cards'] = []
                
            selected_cards = current_player.get_play(game_state)
            turns_played += 1
            
            if not selected_cards:
                if not game_state['table_eval']:
                    selected_cards = [current_player.hand[0]]
                else:
                    current_idx = 1 - current_idx
                    continue
                    
            curr_eval = evaluate_play(selected_cards)
            current_player.remove_cards(selected_cards)
            
            game_state['table_eval'] = curr_eval
            game_state['table_cards'] = selected_cards
            game_state['is_first_turn'] = False
            last_player_idx = current_idx
                
            if not current_player.hand:
                winner_idx = current_idx
                break
                
            current_idx = 1 - current_idx

        # --- AFTER GAME: CALCULATE REWARDS AND TRAIN ---
        if winner_idx == 0:
            metrics['p1_wins'] += 1
            p1_reward, p2_reward = 1.0, -1.0
        else:
            metrics['p2_wins'] += 1
            p1_reward, p2_reward = -1.0, 1.0

        metrics['game_lengths'].append(turns_played)

        # Compile training data
        train_x = []
        train_y = []

        for state_action in bot1.episode_memory:
            train_x.append(state_action)
            train_y.append(p1_reward) # Target is the realized reward

        for state_action in bot2.episode_memory:
            train_x.append(state_action)
            train_y.append(p2_reward)

        if train_x:
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            
            # Train the network on this game's data
            history = shared_model.fit(train_x, train_y, epochs=1, verbose=0, batch_size=batch_size)
            metrics['losses'].append(history.history['loss'][0])

        # --- PRINT METRICS EVERY 50 EPISODES ---
        if episode % 50 == 0:
            avg_loss = np.mean(metrics['losses'][-50:]) if metrics['losses'] else 0.0
            avg_len = np.mean(metrics['game_lengths'][-50:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode}/{episodes} | Time: {elapsed:.1f}s")
            print(f"  -> Win Rate: P1 ({metrics['p1_wins']}) vs P2 ({metrics['p2_wins']})")
            print(f"  -> Avg Game Length: {avg_len:.1f} turns")
            print(f"  -> Avg Neural Net Loss (MSE): {avg_loss:.4f}")
            print("-" * 50)
            
            # Reset metrics for the next batch
            metrics['p1_wins'] = 0
            metrics['p2_wins'] = 0

    # Save the trained weights so they can be loaded by the UI/Simulations
    print("\nTraining Complete! Saving weights to 'tf_advantage_net.weights.h5'...")
    shared_model.save_weights("tf_advantage_net.weights.h5")

if __name__ == "__main__":
    # You can change episodes to 10,000+ for serious training overnight
    train_self_play(episodes=500)