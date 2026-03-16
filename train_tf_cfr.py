import os
import time
import random
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from helper import RANKS, SUITS, get_card_value, evaluate_play
from tf_deep_cfr_bot import TFDeepCFRBot, create_advantage_network, create_policy_network
from test_tf_cfr import test_model

def worker_generate_batch(num_games, adv_path, pol_path, current_episode, total_episodes):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # --- EXPLORATION DECAY WITH FLOOR ---
    progress = min(1.0, current_episode / (total_episodes * 0.8))
    current_exploration = max(0.02, 0.15 - (progress * 0.13))

    bot1 = TFDeepCFRBot("Player 1", adv_model_path=adv_path, policy_model_path=pol_path, is_training=True, exploration_rate=current_exploration)
    bot2 = TFDeepCFRBot("Player 2", adv_model_path=adv_path, policy_model_path=pol_path, is_training=True, exploration_rate=current_exploration)
    players = [bot1, bot2]

    accumulated_train_x, accumulated_train_y = [], []
    policy_x, policy_y = [], []
    p1_wins, p2_wins = 0, 0
    game_lengths = []

    for _ in range(num_games):
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

        if winner_idx == 0:
            p1_wins += 1
            p1_reward, p2_reward = 1.0, -1.0
        else:
            p2_wins += 1
            p1_reward, p2_reward = -1.0, 1.0

        game_lengths.append(turns_played)

        # --- TRUE REGRET CALCULATION ---
        for step in bot1.episode_memory:
            regret = p1_reward - step['baseline_value']
            accumulated_train_x.append(step['inputs'][step['chosen_index']])
            accumulated_train_y.append(regret)

        for step in bot2.episode_memory:
            regret = p2_reward - step['baseline_value']
            accumulated_train_x.append(step['inputs'][step['chosen_index']])
            accumulated_train_y.append(regret)
            
        for mem in bot1.policy_memory + bot2.policy_memory:
            policy_x.append(mem[0])
            policy_y.append(mem[1])

    return accumulated_train_x, accumulated_train_y, policy_x, policy_y, p1_wins, p2_wins, game_lengths

def train_self_play(total_episodes=2000000, batch_size=1024, adv_path='tf_advantage_net.weights.h5', pol_path='tf_policy_net.weights.h5', num_workers=None, episodes_per_update=1000):
    print("="*60)
    print("INITIALIZING AUTHENTIC DEEP CFR (DUAL NETWORK)")
    print("="*60)
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    print(f"Server Architecture: 1 Learner (GPU/Main) | {num_workers} Actors (CPU Workers)")

    # --- LEARNING RATE DECAY ---
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.96, staircase=True)
    optimizer_adv = optimizers.Adam(learning_rate=lr_schedule)
    optimizer_pol = optimizers.Adam(learning_rate=lr_schedule)

    shared_adv_net = create_advantage_network()
    shared_pol_net = create_policy_network()
    
    try:
        shared_adv_net.load_weights(adv_path)
        shared_pol_net.load_weights(pol_path)
        print("Loaded existing weights for both networks.")
    except:
        print("Starting with fresh network weights.")

    shared_adv_net.compile(optimizer=optimizer_adv, loss='mse')
    shared_pol_net.compile(optimizer=optimizer_pol, loss='mse') 

    shared_adv_net.save_weights(adv_path)
    shared_pol_net.save_weights(pol_path)

    metrics = {'p1_wins': 0, 'p2_wins': 0, 'game_lengths': [], 'adv_losses': [], 'pol_losses': []}
    start_time = time.time()
    ctx = mp.get_context('spawn')
    episodes_completed = 0

    MAX_BUFFER_SIZE = 200000
    replay_buffer_x, replay_buffer_y = [], []
    policy_buffer_x, policy_buffer_y = [], []
    total_games_generated = 0

    while episodes_completed < total_episodes:
        base_games = episodes_per_update // num_workers
        remainder = episodes_per_update % num_workers
        worker_tasks = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]

        acc_new_adv_x, acc_new_adv_y = [], []
        acc_new_pol_x, acc_new_pol_y = [], []

        with ctx.Pool(processes=num_workers) as pool:
            results = [pool.apply_async(worker_generate_batch, args=(games, adv_path, pol_path, episodes_completed, total_episodes)) 
                       for games in worker_tasks if games > 0]
            
            for r in results:
                adv_x, adv_y, pol_x, pol_y, p1w, p2w, gl = r.get()
                
                acc_new_adv_x.extend(adv_x)
                acc_new_adv_y.extend(adv_y)
                acc_new_pol_x.extend(pol_x)
                acc_new_pol_y.extend(pol_y)
                
                metrics['p1_wins'] += p1w
                metrics['p2_wins'] += p2w
                metrics['game_lengths'].extend(gl)

                for i in range(len(adv_x)):
                    total_games_generated += 1
                    if len(replay_buffer_x) < MAX_BUFFER_SIZE:
                        replay_buffer_x.append(adv_x[i])
                        replay_buffer_y.append(adv_y[i])
                        policy_buffer_x.append(pol_x[i])
                        policy_buffer_y.append(pol_y[i])
                    else:
                        replace_idx = random.randint(0, total_games_generated - 1)
                        if replace_idx < MAX_BUFFER_SIZE:
                            replay_buffer_x[replace_idx] = adv_x[i]
                            replay_buffer_y[replace_idx] = adv_y[i]
                            policy_buffer_x[replace_idx] = pol_x[i]
                            policy_buffer_y[replace_idx] = pol_y[i]

        if replay_buffer_x and policy_buffer_x:
            target_batch_size = 32000
            
            # --- 1. TRAIN ADVANTAGE NETWORK ---
            train_adv_x_list = list(acc_new_adv_x)
            train_adv_y_list = list(acc_new_adv_y)
            hist_needed_adv = min(target_batch_size - len(train_adv_x_list), len(replay_buffer_x))
            
            if hist_needed_adv > 0:
                indices = np.random.choice(len(replay_buffer_x), hist_needed_adv, replace=False)
                for idx in indices:
                    train_adv_x_list.append(replay_buffer_x[idx])
                    train_adv_y_list.append(replay_buffer_y[idx])
            
            train_adv_x = np.array(train_adv_x_list)
            train_adv_y = np.array(train_adv_y_list)
            shuff_adv = np.arange(len(train_adv_x))
            np.random.shuffle(shuff_adv)
            
            h1 = shared_adv_net.fit(train_adv_x[shuff_adv], train_adv_y[shuff_adv], epochs=1, verbose=0, batch_size=batch_size)
            metrics['adv_losses'].append(h1.history['loss'][0])
            
            # --- 2. TRAIN AVERAGE POLICY NETWORK ---
            train_pol_x_list = list(acc_new_pol_x)
            train_pol_y_list = list(acc_new_pol_y)
            hist_needed_pol = min(target_batch_size - len(train_pol_x_list), len(policy_buffer_x))
            
            if hist_needed_pol > 0:
                indices = np.random.choice(len(policy_buffer_x), hist_needed_pol, replace=False)
                for idx in indices:
                    train_pol_x_list.append(policy_buffer_x[idx])
                    train_pol_y_list.append(policy_buffer_y[idx])
            
            train_pol_x = np.array(train_pol_x_list)
            train_pol_y = np.array(train_pol_y_list)
            shuff_pol = np.arange(len(train_pol_x))
            np.random.shuffle(shuff_pol)
            
            h2 = shared_pol_net.fit(train_pol_x[shuff_pol], train_pol_y[shuff_pol], epochs=1, verbose=0, batch_size=batch_size)
            metrics['pol_losses'].append(h2.history['loss'][0])
            
            # Save files
            shared_adv_net.save_weights(adv_path)
            shared_pol_net.save_weights(pol_path)

            episodes_completed += episodes_per_update

            # Benchmark every 10 updates (10,000 games) using 500 games for smooth curves
            if episodes_completed % (episodes_per_update * 10) == 0:
                test_model(num_games=500, policy_path=pol_path)

        # Print Analytics
        avg_adv = metrics['adv_losses'][-1] if metrics['adv_losses'] else 0.0
        avg_pol = metrics['pol_losses'][-1] if metrics['pol_losses'] else 0.0
        avg_len = np.mean(metrics['game_lengths'][-episodes_per_update:])
        elapsed = time.time() - start_time
        
        print(f"Episodes {episodes_completed}/{total_episodes} | Time: {elapsed:.1f}s")
        print(f"  -> Win Rate: P1 ({metrics['p1_wins']}) vs P2 ({metrics['p2_wins']})")
        print(f"  -> Avg Game Length: {avg_len:.1f} turns")
        print(f"  -> Buffer Size: {len(replay_buffer_x)} / {MAX_BUFFER_SIZE}")
        print(f"  -> Adv Loss: {avg_adv:.4f} | Pol Loss: {avg_pol:.4f}")
        print("-" * 60)
        
        metrics['p1_wins'] = 0
        metrics['p2_wins'] = 0

    print(f"\nTraining Complete! Final weights secured.")

if __name__ == "__main__":
    mp.freeze_support()
    train_self_play(total_episodes=2000000, episodes_per_update=1000, batch_size=1024)
