import os
import time
import random
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from tensorflow.keras import optimizers

from helper import RANKS, SUITS, get_card_value, evaluate_play
from tf_deep_cfr_bot import TFDeepCFRBot, create_advantage_network
from test_tf_cfr import test_model

# ==============================================================================
# THE CLIENT (ACTOR WORKER)
# This function runs isolated in parallel sub-processes. We force it to use 
# the CPU so that parallel workers don't fight over GPU VRAM.
# ==============================================================================
def worker_generate_batch(num_games, model_path):
    # Force workers to use CPU for generation to prevent GPU OOM crashes
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Suppress verbose TF logging in workers
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # Initialize isolated bots for this specific process
    bot1 = TFDeepCFRBot("Player 1", model_path=model_path)
    bot2 = TFDeepCFRBot("Player 2", model_path=model_path)
    players = [bot1, bot2]

    accumulated_train_x = []
    accumulated_train_y = []
    p1_wins = 0
    p2_wins = 0
    game_lengths = []

    for _ in range(num_games):
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
            'dead_cards': []
        }
        
        last_player_idx = None
        turns_played = 0

        # --- PLAY A SINGLE GAME ---
        while True:
            current_player = players[current_idx]
            
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
                game_state['dead_cards'].extend(game_state['table_cards'])
                break
                
            current_idx = 1 - current_idx

        # --- AFTER GAME: CALCULATE REWARDS ---
        if winner_idx == 0:
            p1_wins += 1
            p1_reward, p2_reward = 1.0, -1.0
        else:
            p2_wins += 1
            p1_reward, p2_reward = -1.0, 1.0

        game_lengths.append(turns_played)

        # Append this episode's experience
        for state_action in bot1.episode_memory:
            accumulated_train_x.append(state_action)
            accumulated_train_y.append(p1_reward)

        for state_action in bot2.episode_memory:
            accumulated_train_x.append(state_action)
            accumulated_train_y.append(p2_reward)

    return accumulated_train_x, accumulated_train_y, p1_wins, p2_wins, game_lengths


# ==============================================================================
# THE SERVER (LEARNER MASTER)
# Orchestrates workers, gathers data, and trains on the GPU.
# ==============================================================================
def train_self_play(total_episodes=2000, batch_size=64, model_path='tf_advantage_net.weights.h5', num_workers=None, episodes_per_update=100):
    print("="*60)
    print("INITIALIZING DISTRIBUTED DEEP CFR (TENSORFLOW)")
    print("="*60)
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave 1 core for the master process
        
    print(f"Server Architecture: 1 Learner (GPU/Main) | {num_workers} Actors (CPU Workers)")

    # 1. Create the Master Neural Network
    shared_model = create_advantage_network()
    try:
        shared_model.load_weights(model_path)
        print(f"Loaded existing weights from '{model_path}'")
    except (FileNotFoundError, OSError):
        print(f"No existing weights found at '{model_path}', starting with random weights.")

    optimizer = optimizers.Adam(learning_rate=1e-4)
    shared_model.compile(optimizer=optimizer, loss='mse')

    # Save initial weights immediately so workers have a file to load
    shared_model.save_weights(model_path)

    metrics = {'p1_wins': 0, 'p2_wins': 0, 'game_lengths': [], 'losses': []}
    start_time = time.time()

    # Use 'spawn' to ensure TensorFlow state is not corrupted during process fork
    ctx = mp.get_context('spawn')
    
    episodes_completed = 0

    # 2. Main Distributed Loop
    while episodes_completed < total_episodes:
        # Calculate how to distribute the games across our CPU workers
        base_games = episodes_per_update // num_workers
        remainder = episodes_per_update % num_workers
        worker_tasks = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]

        accumulated_train_x = []
        accumulated_train_y = []

        # Dispatch Tasks to Actors
        with ctx.Pool(processes=num_workers) as pool:
            # We use apply_async to fire them off in parallel
            results = [pool.apply_async(worker_generate_batch, args=(games, model_path)) 
                       for games in worker_tasks if games > 0]
            
            # Wait and harvest data
            for r in results:
                x, y, p1w, p2w, gl = r.get()
                accumulated_train_x.extend(x)
                accumulated_train_y.extend(y)
                metrics['p1_wins'] += p1w
                metrics['p2_wins'] += p2w
                metrics['game_lengths'].extend(gl)

        # 3. Master Training Node (Learner)
        if accumulated_train_x:
            train_x = np.array(accumulated_train_x)
            train_y = np.array(accumulated_train_y)
            
            # Shuffle batch
            indices = np.arange(len(train_x))
            np.random.shuffle(indices)
            train_x = train_x[indices]
            train_y = train_y[indices]

            # Fit the model
            history = shared_model.fit(train_x, train_y, epochs=2, verbose=0, batch_size=batch_size)
            metrics['losses'].append(history.history['loss'][0])
            
            # Flush updated weights to disk so the next round of actors pick them up!
            shared_model.save_weights(model_path)

            # Test
            test_model(num_games=1000)

        episodes_completed += episodes_per_update

        # Print Analytics
        avg_loss = metrics['losses'][-1] if metrics['losses'] else 0.0
        avg_len = np.mean(metrics['game_lengths'][-episodes_per_update:])
        elapsed = time.time() - start_time
        
        print(f"Episodes {episodes_completed}/{total_episodes} | Time: {elapsed:.1f}s")
        print(f"  -> Win Rate: P1 ({metrics['p1_wins']}) vs P2 ({metrics['p2_wins']})")
        print(f"  -> Avg Game Length: {avg_len:.1f} turns")
        print(f"  -> Train Batch Size: {len(train_x)} samples")
        print(f"  -> Neural Net Loss (MSE): {avg_loss:.4f}")
        print("-" * 60)
        
        # Reset counters for the next batch
        metrics['p1_wins'] = 0
        metrics['p2_wins'] = 0

    print(f"\nTraining Complete! Final weights secured at '{model_path}'...")

if __name__ == "__main__":
    # Ensure Windows/Mac compatibility with multiprocessing
    mp.freeze_support()
    train_self_play(total_episodes=2000000, episodes_per_update=10000, batch_size=1024)