import os
import time
import random
import numpy as np
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from helper import RANKS, SUITS, get_card_value, evaluate_play
from player import BotPlayer
from tf_deep_cfr_bot import TFDeepCFRBot, create_advantage_network, create_policy_network
from test_tf_cfr import test_model

# ==============================================================================
# THE GPU INFERENCE SERVER
# ==============================================================================
def gpu_inference_server(conns, adv_net, pol_net):
    active_conns = list(conns)
    while active_conns:
        adv_reqs, adv_pipes, adv_lens = [], [], []
        pol_reqs, pol_pipes, pol_lens = [], [], []
        processed_any = False
        
        for conn in active_conns[:]: 
            if conn.poll(): 
                processed_any = True
                try:
                    msg = conn.recv()
                    if msg == "DONE":
                        active_conns.remove(conn)
                    else:
                        is_policy, inputs = msg
                        if is_policy:
                            pol_reqs.append(inputs)
                            pol_pipes.append(conn)
                            pol_lens.append(len(inputs))
                        else:
                            adv_reqs.append(inputs)
                            adv_pipes.append(conn)
                            adv_lens.append(len(inputs))
                except EOFError:
                    active_conns.remove(conn)
        
        if not processed_any:
            time.sleep(0.001)
            continue

        if adv_reqs:
            batch = np.concatenate(adv_reqs, axis=0)
            preds = adv_net(tf.convert_to_tensor(batch), training=False).numpy()
            idx = 0
            for c, length in zip(adv_pipes, adv_lens):
                c.send(preds[idx : idx + length])
                idx += length
                
        if pol_reqs:
            batch = np.concatenate(pol_reqs, axis=0)
            preds = pol_net(tf.convert_to_tensor(batch), training=False).numpy()
            idx = 0
            for c, length in zip(pol_pipes, pol_lens):
                c.send(preds[idx : idx + length])
                idx += length

# ==============================================================================
# INTRA-PROCESS THREAD LOGIC (GAME SIMULATION)
# ==============================================================================
def _thread_simulate_games(num_games, conn, current_episode, total_episodes):
    """The isolated game loop executed by individual threads."""
    progress = min(1.0, current_episode / (total_episodes * 0.8))
    current_exploration = max(0.02, 0.15 - (progress * 0.13))

    bot1 = TFDeepCFRBot("Player 1", pipe=conn, is_training=True, exploration_rate=current_exploration)
    bot2 = TFDeepCFRBot("Player 2", pipe=conn, is_training=True, exploration_rate=current_exploration)
    players = [bot1, bot2]

    accumulated_train_x, accumulated_train_y = [], []
    policy_x, policy_y = [], []
    p1_wins, p2_wins = 0, 0
    game_lengths = []

    for _ in range(num_games):
        deck = [(rank, suit) for rank in RANKS for suit in SUITS]
        random.shuffle(deck)
        piles = [deck[i * 13 : (i + 1) * 13] for i in range(4)]
        
        bot1.hand, bot2.hand = [], []
        bot1.receive_cards(piles[0])
        bot2.receive_cards(piles[1])
        bot1.clear_memory()
        bot2.clear_memory()

        current_idx = 0 if get_card_value(bot1.hand[0]) < get_card_value(bot2.hand[0]) else 1
        lowest_card = bot1.hand[0] if current_idx == 0 else bot2.hand[0]

        game_state = { 'table_eval': None, 'table_cards': [], 'is_first_turn': True, 'lowest_card': lowest_card, 'dead_cards': [] }
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
                if not game_state['table_eval']: selected_cards = [current_player.hand[0]]
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

        for step in bot1.episode_memory:
            accumulated_train_x.append(step['inputs'][step['chosen_index']])
            accumulated_train_y.append(p1_reward - step['baseline_value'])

        for step in bot2.episode_memory:
            accumulated_train_x.append(step['inputs'][step['chosen_index']])
            accumulated_train_y.append(p2_reward - step['baseline_value'])
            
        for mem in bot1.policy_memory + bot2.policy_memory:
            policy_x.append(mem[0])
            policy_y.append(mem[1])

    return accumulated_train_x, accumulated_train_y, policy_x, policy_y, p1_wins, p2_wins, game_lengths


# ==============================================================================
# CPU MULTIPROCESS MANAGERS
# ==============================================================================
def worker_generate_batch(num_games, conns, result_queue, current_episode, total_episodes):
    """Process Manager: Spawns threads to overlap Python processing with GPU Wait times."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    threads_count = len(conns)
    base_games = num_games // threads_count
    remainder = num_games % threads_count
    thread_tasks = [base_games + (1 if i < remainder else 0) for i in range(threads_count)]

    results = []
    # Utilize Python threads to hide IPC and GPU latency
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        futures = []
        for i in range(threads_count):
            if thread_tasks[i] > 0:
                futures.append(executor.submit(_thread_simulate_games, thread_tasks[i], conns[i], current_episode, total_episodes))
        
        for future in futures:
            results.append(future.result())

    # Aggregate thread results
    accumulated_train_x, accumulated_train_y = [], []
    policy_x, policy_y = [], []
    p1_wins, p2_wins = 0, 0
    game_lengths = []

    for r in results:
        adv_x, adv_y, pol_x, pol_y, p1w, p2w, gl = r
        accumulated_train_x.extend(adv_x)
        accumulated_train_y.extend(adv_y)
        policy_x.extend(pol_x)
        policy_y.extend(pol_y)
        p1_wins += p1w
        p2_wins += p2w
        game_lengths.extend(gl)

    for conn in conns:
        conn.send("DONE") 

    result_queue.put((accumulated_train_x, accumulated_train_y, policy_x, policy_y, p1_wins, p2_wins, game_lengths))

# ==============================================================================
# MAIN GPU LEARNER NODE
# ==============================================================================
def train_self_play(total_episodes=2000000, batch_size=1024, adv_path='tf_advantage_net.weights.h5', pol_path='tf_policy_net.weights.h5', num_workers=None, threads_per_worker=4, episodes_per_update=1000):
    print("="*60)
    print("INITIALIZING PIPED GPU-INFERENCE CFR (MULTI-THREADED)")
    print("="*60)
    
    if num_workers is None: num_workers = max(1, mp.cpu_count() - 1)
    total_threads = num_workers * threads_per_worker
    print(f"Server Architecture: 1 GPU Predictor | {num_workers} Processes x {threads_per_worker} Threads ({total_threads} virtual actors)")

    lr_schedule = ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.96, staircase=True)
    optimizer_adv = optimizers.Adam(learning_rate=lr_schedule)
    optimizer_pol = optimizers.Adam(learning_rate=lr_schedule)

    shared_adv_net = create_advantage_network()
    shared_pol_net = create_policy_network()
    
    try:
        shared_adv_net.load_weights(adv_path)
        shared_pol_net.load_weights(pol_path)
        print("Loaded existing weights for both networks.")
    except: print("Starting with fresh network weights.")

    shared_adv_net.compile(optimizer=optimizer_adv, loss='mse')
    shared_pol_net.compile(optimizer=optimizer_pol, loss='mse') 

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

        acc_new_adv_x, acc_new_adv_y, acc_new_pol_x, acc_new_pol_y = [], [], [], []

        # Create a massive bank of pipes (1 for every thread)
        pipes = [ctx.Pipe() for _ in range(total_threads)]
        parent_conns = [p[0] for p in pipes]
        child_conns = [p[1] for p in pipes]
        
        # Group the child pipes into chunks for the processes
        child_conn_chunks = [child_conns[i * threads_per_worker : (i + 1) * threads_per_worker] for i in range(num_workers)]
        
        result_queue = ctx.Queue()

        processes = []
        for i, task_count in enumerate(worker_tasks):
            if task_count > 0:
                p = ctx.Process(target=worker_generate_batch, args=(task_count, child_conn_chunks[i], result_queue, episodes_completed, total_episodes))
                p.start()
                processes.append(p)

        gpu_inference_server(parent_conns, shared_adv_net, shared_pol_net)

        for _ in range(len(processes)):
            adv_x, adv_y, pol_x, pol_y, p1w, p2w, gl = result_queue.get()
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

        for p in processes: p.join()

        if replay_buffer_x and policy_buffer_x:
            target_batch_size = 32000
            
            # ADVANTAGE TRAINING (LEI)
            train_adv_x_list, train_adv_y_list = list(acc_new_adv_x), list(acc_new_adv_y)
            hist_needed_adv = min(target_batch_size - len(train_adv_x_list), len(replay_buffer_x))
            if hist_needed_adv > 0:
                indices = np.random.choice(len(replay_buffer_x), hist_needed_adv, replace=False)
                for idx in indices:
                    train_adv_x_list.append(replay_buffer_x[idx])
                    train_adv_y_list.append(replay_buffer_y[idx])
            
            train_adv_x, train_adv_y = np.array(train_adv_x_list), np.array(train_adv_y_list)
            shuff_adv = np.arange(len(train_adv_x))
            np.random.shuffle(shuff_adv)
            
            h1 = shared_adv_net.fit(train_adv_x[shuff_adv], train_adv_y[shuff_adv], epochs=1, verbose=0, batch_size=batch_size)
            metrics['adv_losses'].append(h1.history['loss'][0])
            
            # POLICY TRAINING (LEI)
            train_pol_x_list, train_pol_y_list = list(acc_new_pol_x), list(acc_new_pol_y)
            hist_needed_pol = min(target_batch_size - len(train_pol_x_list), len(policy_buffer_x))
            if hist_needed_pol > 0:
                indices = np.random.choice(len(policy_buffer_x), hist_needed_pol, replace=False)
                for idx in indices:
                    train_pol_x_list.append(policy_buffer_x[idx])
                    train_pol_y_list.append(policy_buffer_y[idx])
            
            train_pol_x, train_pol_y = np.array(train_pol_x_list), np.array(train_pol_y_list)
            shuff_pol = np.arange(len(train_pol_x))
            np.random.shuffle(shuff_pol)
            
            h2 = shared_pol_net.fit(train_pol_x[shuff_pol], train_pol_y[shuff_pol], epochs=1, verbose=0, batch_size=batch_size)
            metrics['pol_losses'].append(h2.history['loss'][0])
            
            shared_adv_net.save_weights(adv_path)
            shared_pol_net.save_weights(pol_path)

            episodes_completed += episodes_per_update

        avg_adv = metrics['adv_losses'][-1] if metrics['adv_losses'] else 0.0
        avg_pol = metrics['pol_losses'][-1] if metrics['pol_losses'] else 0.0
        avg_len = np.mean(metrics['game_lengths'][-episodes_per_update:])
        test_model(num_games=1000, threads_per_worker=100)
        elapsed = time.time() - start_time
        
        print(f"Episodes {episodes_completed}/{total_episodes} | Time: {elapsed:.1f}s")
        print(f"  -> Win Rate: P1 ({metrics['p1_wins']}) vs P2 ({metrics['p2_wins']})")
        print(f"  -> Avg Game Length: {avg_len:.1f} turns")
        print(f"  -> Buffer Size: {len(replay_buffer_x)} / {MAX_BUFFER_SIZE}")
        print(f"  -> Adv Loss: {avg_adv:.4f} | Pol Loss: {avg_pol:.4f}")
        print("-" * 60)
        metrics['p1_wins'], metrics['p2_wins'] = 0, 0

    print(f"\nTraining Complete! Final weights secured.")

if __name__ == "__main__":
    mp.freeze_support()
    # threads_per_worker parameter controls how many overlapping threads spawn per core
    train_self_play(total_episodes=2000000, episodes_per_update=1000, batch_size=1024, threads_per_worker=100)