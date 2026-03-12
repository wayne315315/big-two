import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import time

from helper import RANKS, SUITS, get_card_value, evaluate_play
from tf_deep_cfr_bot import TFDeepCFRBot, create_advantage_network

def train_self_play(episodes=1000, batch_size=64, model_path='tf_advantage_net.weights.h5'):
    print("="*50)
    print("INITIALIZING DEEP CFR SELF-PLAY TRAINING (TENSORFLOW)")
    print("="*50)

    # 1. Create the shared Neural Network and the optimizer
    shared_model = create_advantage_network()
    try:
        shared_model.load_weights(model_path)  # Load existing weights if available
        print(f"Loaded existing weights from '{model_path}'")
    except (FileNotFoundError, OSError):
        print(f"No existing weights found at '{model_path}', starting with random weights.")

    optimizer = optimizers.Adam(learning_rate=0.001)
    shared_model.compile(optimizer=optimizer, loss='mse')

    # 2. Setup the training bots (Directly passing the shared model)
    bot1 = TFDeepCFRBot("Player 1", model=shared_model)
    bot2 = TFDeepCFRBot("Player 2", model=shared_model)
    players = [bot1, bot2]

    metrics = {'p1_wins': 0, 'p2_wins': 0, 'game_lengths': [], 'losses': []}
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
            'lowest_card': lowest_card,
            'dead_cards': []  # <--- CORRECTED: Starts strictly empty!
        }
        
        last_player_idx = None
        turns_played = 0

        # --- PLAY A SINGLE GAME ---
        while True:
            current_player = players[current_idx]
            
            # If the trick is won, the LAST remaining table cards are swept into the dead/discard pile
            if last_player_idx == current_idx:
                game_state['dead_cards'].extend(game_state['table_cards'])
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
            # Before overwriting the table, move the beaten cards to the dead pile
            game_state['dead_cards'].extend(game_state['table_cards'])
            
            game_state['table_eval'] = curr_eval
            game_state['table_cards'] = selected_cards
            game_state['is_first_turn'] = False
            last_player_idx = current_idx
                
            if not current_player.hand:
                winner_idx = current_idx
                # Sweep the final winning play into the dead cards just for completion
                game_state['dead_cards'].extend(game_state['table_cards'])
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

        train_x = []
        train_y = []

        for state_action in bot1.episode_memory:
            train_x.append(state_action)
            train_y.append(p1_reward)

        for state_action in bot2.episode_memory:
            train_x.append(state_action)
            train_y.append(p2_reward)

        if train_x:
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            history = shared_model.fit(train_x, train_y, epochs=1, verbose=0, batch_size=batch_size)
            metrics['losses'].append(history.history['loss'][0])

        if episode % 50 == 0:
            avg_loss = np.mean(metrics['losses'][-50:]) if metrics['losses'] else 0.0
            avg_len = np.mean(metrics['game_lengths'][-50:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode}/{episodes} | Time: {elapsed:.1f}s")
            print(f"  -> Win Rate: P1 ({metrics['p1_wins']}) vs P2 ({metrics['p2_wins']})")
            print(f"  -> Avg Game Length: {avg_len:.1f} turns")
            print(f"  -> Avg Neural Net Loss (MSE): {avg_loss:.4f}")
            print("-" * 50)
            metrics['p1_wins'] = 0
            metrics['p2_wins'] = 0

    print(f"\nTraining Complete! Saving weights to '{model_path}'...")
    shared_model.save_weights(model_path)

if __name__ == "__main__":
    train_self_play(episodes=500)