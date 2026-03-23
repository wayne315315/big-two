import os
import time
import random
import pickle
import numpy as np
import multiprocessing as mp
import multiprocessing.connection 
import threading
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras import optimizers, mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from helper import RANKS, SUITS, get_card_value, evaluate_play
from player import BotPlayer
from tf_deep_cfr_bot import TFDeepCFRBot, create_advantage_network, create_policy_network
from test_tf_cfr import test_model

def add_to_buffer(buffer_x, buffer_y, new_x, new_y, current_size, max_size):
    n = len(new_x)
    if n == 0: return current_size
    if current_size + n <= max_size:
        buffer_x[current_size:current_size+n] = new_x
        buffer_y[current_size:current_size+n] = new_y
        return current_size + n
    else:
        rem = max_size - current_size
        if rem > 0:
            buffer_x[current_size:max_size] = new_x[:rem]
            buffer_y[current_size:max_size] = new_y[:rem]
        n_left = n - rem
        if n_left > 0:
            replace_idx = np.random.randint(0, max_size, size=n_left)
            buffer_x[replace_idx] = new_x[rem:]
            buffer_y[replace_idx] = new_y[rem:]
        return max_size

def gpu_inference_server(conns, fast_adv_infer, fast_pol_infer):
    FIXED_BATCH = 16384  
    active_conns = list(conns)
    
    MIN_BATCH_SIZE = 512
    MAX_WAIT_TIME = 0.01 

    adv_buffer = np.zeros((FIXED_BATCH, 37, 55), dtype=np.float16)
    pol_buffer = np.zeros((FIXED_BATCH, 37, 55), dtype=np.float16)
    
    adv_pipes, adv_lens = [], []
    pol_pipes, pol_lens = [], []
    
    adv_cursor = 0
    pol_cursor = 0
    last_fire_time = time.time()

    while active_conns:
        ready_conns = []
        for i in range(0, len(active_conns), 500):
            chunk = active_conns[i : i+500]
            ready_conns.extend(multiprocessing.connection.wait(chunk, timeout=0.001))
            
        for conn in ready_conns: 
            try:
                msg = conn.recv()
                if msg == "DONE":
                    active_conns.remove(conn)
                else:
                    is_policy, inputs = msg
                    length = len(inputs)
                    
                    if is_policy:
                        if pol_cursor + length > FIXED_BATCH:
                            tensor = tf.convert_to_tensor(pol_buffer, dtype=tf.float16)
                            preds = fast_pol_infer(tensor).numpy()
                            idx = 0
                            for c, l in zip(pol_pipes, pol_lens):
                                c.send(preds[idx : idx + l])
                                idx += l
                            pol_pipes, pol_lens = [], []
                            pol_cursor = 0
                            last_fire_time = time.time()
                            
                        pol_buffer[pol_cursor : pol_cursor + length] = inputs
                        pol_pipes.append(conn)
                        pol_lens.append(length)
                        pol_cursor += length
                    else:
                        if adv_cursor + length > FIXED_BATCH:
                            tensor = tf.convert_to_tensor(adv_buffer, dtype=tf.float16)
                            preds = fast_adv_infer(tensor).numpy()
                            idx = 0
                            for c, l in zip(adv_pipes, adv_lens):
                                c.send(preds[idx : idx + l])
                                idx += l
                            adv_pipes, adv_lens = [], []
                            adv_cursor = 0
                            last_fire_time = time.time()
                            
                        adv_buffer[adv_cursor : adv_cursor + length] = inputs
                        adv_pipes.append(conn)
                        adv_lens.append(length)
                        adv_cursor += length
            except EOFError:
                if conn in active_conns:
                    active_conns.remove(conn)

        time_waiting = time.time() - last_fire_time

        if adv_cursor >= MIN_BATCH_SIZE or (time_waiting > MAX_WAIT_TIME and adv_cursor > 0):
            tensor = tf.convert_to_tensor(adv_buffer, dtype=tf.float16)
            preds = fast_adv_infer(tensor).numpy()
            idx = 0
            for c, length in zip(adv_pipes, adv_lens):
                c.send(preds[idx : idx + length])
                idx += length
            adv_pipes, adv_lens = [], []
            adv_cursor = 0
            last_fire_time = time.time()
                
        if pol_cursor >= MIN_BATCH_SIZE or (time_waiting > MAX_WAIT_TIME and pol_cursor > 0):
            tensor = tf.convert_to_tensor(pol_buffer, dtype=tf.float16)
            preds = fast_pol_infer(tensor).numpy()
            idx = 0
            for c, length in zip(pol_pipes, pol_lens):
                c.send(preds[idx : idx + length])
                idx += length
            pol_pipes, pol_lens = [], []
            pol_cursor = 0
            last_fire_time = time.time()

def _thread_simulate_games(num_games, conn, current_episode, total_episodes):
    #progress = min(1.0, current_episode / (total_episodes * 0.8))
    progress = min(1.0, current_episode / (2000000 * 0.8))
    current_exploration = max(0.02, 0.15 - (progress * 0.13))

    # PURE SELF PLAY
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

        game_state = { 
            'table_eval': None, 'table_cards': [], 'is_first_turn': True, 
            'lowest_card': lowest_card, 'dead_cards': [], 'history': [] 
        }
        last_player_idx = None
        turns_played = 0

        while True:
            current_player = players[current_idx]
            
            game_state['my_idx'] = current_idx
            game_state['my_hand_size'] = len(current_player.hand)
            game_state['opp_hand_size'] = len(players[1 - current_idx].hand)

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
                    game_state['history'].append((current_idx, [])) 
                    current_idx = 1 - current_idx
                    continue
                    
            curr_eval = evaluate_play(selected_cards)
            current_player.remove_cards(selected_cards)
            
            game_state['history'].append((current_idx, selected_cards)) 
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
            
        for mem in bot1.policy_memory:
            policy_x.append(mem[0])
            policy_y.append(mem[1])

        for step in bot2.episode_memory:
            accumulated_train_x.append(step['inputs'][step['chosen_index']])
            accumulated_train_y.append(p2_reward - step['baseline_value'])
            
        for mem in bot2.policy_memory:
            policy_x.append(mem[0])
            policy_y.append(mem[1])

    return (
        np.array(accumulated_train_x, dtype=np.float16), 
        np.array(accumulated_train_y, dtype=np.float32), 
        np.array(policy_x, dtype=np.float16), 
        np.array(policy_y, dtype=np.float32), 
        p1_wins, p2_wins, game_lengths
    )

def worker_generate_batch(num_games, conns, result_queue, current_episode, total_episodes):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    threads_count = len(conns)
    base_games = num_games // threads_count
    remainder = num_games % threads_count
    thread_tasks = [base_games + (1 if i < remainder else 0) for i in range(threads_count)]

    results = []
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        futures = []
        for i in range(threads_count):
            if thread_tasks[i] > 0:
                futures.append(executor.submit(_thread_simulate_games, thread_tasks[i], conns[i], current_episode, total_episodes))
        
        for future in futures:
            results.append(future.result())

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

    result_queue.put((
        np.array(accumulated_train_x, dtype=np.float16), 
        np.array(accumulated_train_y, dtype=np.float32), 
        np.array(policy_x, dtype=np.float16), 
        np.array(policy_y, dtype=np.float32), 
        p1_wins, p2_wins, game_lengths
    ))

def train_self_play(total_episodes=2000000, batch_size=4096, adv_path='tf_advantage_model.keras', pol_path='tf_policy_model.keras', buffer_path='cfr_replay_buffers.pkl', num_workers=None, threads_per_worker=30, episodes_per_update=1000):
    print("="*60, flush=True)
    print("INITIALIZING TEMPORAL 1D-RESNET CFR PIPELINE", flush=True)
    print("="*60, flush=True)
    
    if num_workers is None: num_workers = max(1, mp.cpu_count() - 1)
    total_threads = num_workers * threads_per_worker
    print(f"Server Architecture: 1 GPU Predictor | {num_workers} Processes x {threads_per_worker} Threads ({total_threads} virtual actors)", flush=True)

    try:
        shared_adv_net = tf.keras.models.load_model(adv_path)
        shared_pol_net = tf.keras.models.load_model(pol_path)
        print(f"Loaded existing FULL models from .keras files.", flush=True)
    except Exception as e:
        print("Starting with fresh network weights and optimizers.", flush=True)
        shared_adv_net = create_advantage_network()
        shared_pol_net = create_policy_network()
        
    optimizer_adv = optimizers.Adam(learning_rate=1e-4)
    optimizer_pol = optimizers.Adam(learning_rate=1e-4)

    shared_adv_net.compile(optimizer=optimizer_adv, loss='mse', jit_compile=True)
    shared_pol_net.compile(optimizer=optimizer_pol, loss='mse', jit_compile=True) 

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(batch_size, 37, 55), dtype=tf.float16),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
    ], jit_compile=True)
    def train_adv_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            preds = shared_adv_net(x_batch, training=True)
            loss = tf.reduce_mean(tf.square(y_batch - tf.squeeze(preds)))
        grads = tape.gradient(loss, shared_adv_net.trainable_variables)
        optimizer_adv.apply_gradients(zip(grads, shared_adv_net.trainable_variables))
        return loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(batch_size, 37, 55), dtype=tf.float16),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
    ], jit_compile=True)
    def train_pol_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            preds = shared_pol_net(x_batch, training=True)
            loss = tf.reduce_mean(tf.square(y_batch - tf.squeeze(preds)))
        grads = tape.gradient(loss, shared_pol_net.trainable_variables)
        optimizer_pol.apply_gradients(zip(grads, shared_pol_net.trainable_variables))
        return loss

    FIXED_BATCH = 16384
    @tf.function(input_signature=[tf.TensorSpec(shape=(FIXED_BATCH, 37, 55), dtype=tf.float16)], jit_compile=True)
    def fast_adv_infer(batch):
        return shared_adv_net(batch, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=(FIXED_BATCH, 37, 55), dtype=tf.float16)], jit_compile=True)
    def fast_pol_infer(batch):
        return shared_pol_net(batch, training=False)

    print("Compiling XLA Temporal Conv1D Kernels (This takes a few seconds)...", flush=True)
    _ = fast_adv_infer(tf.zeros((FIXED_BATCH, 37, 55), dtype=tf.float16))
    _ = fast_pol_infer(tf.zeros((FIXED_BATCH, 37, 55), dtype=tf.float16))
    _ = train_adv_step(tf.zeros((batch_size, 37, 55), dtype=tf.float16), tf.zeros((batch_size,), dtype=tf.float32))
    _ = train_pol_step(tf.zeros((batch_size, 37, 55), dtype=tf.float16), tf.zeros((batch_size,), dtype=tf.float32))
    print("XLA Compilation Complete! Inference Engine Armed.", flush=True)

    metrics = {'p1_wins': 0, 'p2_wins': 0, 'game_lengths': [], 'adv_losses': [], 'pol_losses': []}
    start_time = time.time()
    ctx = mp.get_context('spawn')
    
    MAX_BUFFER_SIZE = 200000
    replay_adv_x = np.zeros((MAX_BUFFER_SIZE, 37, 55), dtype=np.float16)
    replay_adv_y = np.zeros((MAX_BUFFER_SIZE,), dtype=np.float32)
    replay_pol_x = np.zeros((MAX_BUFFER_SIZE, 37, 55), dtype=np.float16)
    replay_pol_y = np.zeros((MAX_BUFFER_SIZE,), dtype=np.float32)
    
    replay_adv_size = 0
    replay_pol_size = 0
    episodes_completed = 0
    total_games_generated = 0

    if os.path.exists(buffer_path):
        print(f"Restoring historical replay buffers from '{buffer_path}'...", flush=True)
        with open(buffer_path, 'rb') as f:
            save_data = pickle.load(f)
            
            loaded_adv_x = save_data['adv_x']
            replay_adv_size = len(loaded_adv_x)
            replay_adv_x[:replay_adv_size] = loaded_adv_x
            replay_adv_y[:replay_adv_size] = save_data['adv_y']
            
            loaded_pol_x = save_data['pol_x']
            replay_pol_size = len(loaded_pol_x)
            replay_pol_x[:replay_pol_size] = loaded_pol_x
            replay_pol_y[:replay_pol_size] = save_data['pol_y']
            
            episodes_completed = save_data['episodes_completed']
            total_games_generated = save_data['total_games_generated']
        print(f"Successfully restored {replay_adv_size} adv states and {replay_pol_size} pol states.", flush=True)
    else:
        print("No historical buffer found. Starting fresh buffer generation.", flush=True)

    while episodes_completed < total_episodes:
        decay_cycles = episodes_completed // 10000
        current_lr = max(1e-6, 1e-4 * (0.96 ** decay_cycles))
        
        optimizer_adv.learning_rate.assign(current_lr)
        optimizer_pol.learning_rate.assign(current_lr)

        base_games = episodes_per_update // num_workers
        remainder = episodes_per_update % num_workers
        worker_tasks = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]

        pipes = [ctx.Pipe() for _ in range(total_threads)]
        parent_conns = [p[0] for p in pipes]
        child_conns = [p[1] for p in pipes]
        
        child_conn_chunks = [child_conns[i * threads_per_worker : (i + 1) * threads_per_worker] for i in range(num_workers)]
        result_queue = ctx.Queue()

        print("  [Phase 1] Simulation (CPU+GPU Infer)... ", end="", flush=True)
        t_sim_start = time.time()
        
        processes = []
        for i, task_count in enumerate(worker_tasks):
            if task_count > 0:
                p = ctx.Process(target=worker_generate_batch, args=(task_count, child_conn_chunks[i], result_queue, episodes_completed, total_episodes))
                p.start()
                processes.append(p)

        gpu_inference_server(parent_conns, fast_adv_infer, fast_pol_infer)
        
        d_sim = time.time() - t_sim_start
        print(f"Done in {d_sim:.2f}s", flush=True)
        
        print("  [Phase 2] Buffer Collection...          ", end="", flush=True)
        t_col_start = time.time()
        
        for _ in range(len(processes)):
            adv_x, adv_y, pol_x, pol_y, p1w, p2w, gl = result_queue.get()
            metrics['p1_wins'] += p1w
            metrics['p2_wins'] += p2w
            metrics['game_lengths'].extend(gl)
            total_games_generated += len(adv_x)
            
            replay_adv_size = add_to_buffer(replay_adv_x, replay_adv_y, adv_x, adv_y, replay_adv_size, MAX_BUFFER_SIZE)
            replay_pol_size = add_to_buffer(replay_pol_x, replay_pol_y, pol_x, pol_y, replay_pol_size, MAX_BUFFER_SIZE)

        for p in processes: p.join()
        
        d_col = time.time() - t_col_start
        print(f"Done in {d_col:.2f}s", flush=True)

        if replay_adv_size > 0 and replay_pol_size > 0:
            print("  [Phase 3] Training Data Prep...         ", end="", flush=True)
            t_prep_start = time.time()
            
            target_batch_size = 32000
            
            sample_adv_size = min(target_batch_size, replay_adv_size)
            sample_adv_size = (sample_adv_size // batch_size) * batch_size
            if sample_adv_size == 0: sample_adv_size = batch_size
            idx_adv = np.random.choice(replay_adv_size, sample_adv_size, replace=False)
            train_adv_x = replay_adv_x[idx_adv]
            train_adv_y = replay_adv_y[idx_adv]
            
            sample_pol_size = min(target_batch_size, replay_pol_size)
            sample_pol_size = (sample_pol_size // batch_size) * batch_size
            if sample_pol_size == 0: sample_pol_size = batch_size
            idx_pol = np.random.choice(replay_pol_size, sample_pol_size, replace=False)
            train_pol_x = replay_pol_x[idx_pol]
            train_pol_y = replay_pol_y[idx_pol]
            
            d_prep = time.time() - t_prep_start
            print(f"Done in {d_prep:.2f}s", flush=True)
            
            print("  [Phase 4] Model Fitting (GPU Train)...  ", end="", flush=True)
            t_fit_start = time.time()
            
            adv_loss_sum = 0.0
            adv_steps = sample_adv_size // batch_size
            for i in range(0, sample_adv_size, batch_size):
                end = i + batch_size
                loss_a = train_adv_step(train_adv_x[i:end], train_adv_y[i:end])
                adv_loss_sum += float(loss_a)
                
            pol_loss_sum = 0.0
            pol_steps = sample_pol_size // batch_size
            for i in range(0, sample_pol_size, batch_size):
                end = i + batch_size
                loss_p = train_pol_step(train_pol_x[i:end], train_pol_y[i:end])
                pol_loss_sum += float(loss_p)
                
            metrics['adv_losses'].append(adv_loss_sum / max(1, adv_steps))
            metrics['pol_losses'].append(pol_loss_sum / max(1, pol_steps))
            
            d_fit = time.time() - t_fit_start
            print(f"Done in {d_fit:.2f}s", flush=True)
            
            print("  [Phase 5] Disk Saving (I/O)...          ", end="", flush=True)
            t_save_start = time.time()
            
            shared_adv_net.save(adv_path)
            shared_pol_net.save(pol_path)
            episodes_completed += episodes_per_update
            
            if episodes_completed % 5000 == 0:
                def async_save():
                    with open(buffer_path, 'wb') as f:
                        pickle.dump({
                            'adv_x': np.copy(replay_adv_x[:replay_adv_size]),
                            'adv_y': np.copy(replay_adv_y[:replay_adv_size]),
                            'pol_x': np.copy(replay_pol_x[:replay_pol_size]),
                            'pol_y': np.copy(replay_pol_y[:replay_pol_size]),
                            'episodes_completed': episodes_completed,
                            'total_games_generated': total_games_generated
                        }, f, protocol=pickle.HIGHEST_PROTOCOL)
                threading.Thread(target=async_save).start()
                
                d_save = time.time() - t_save_start
                print(f"Done in {d_save:.2f}s (Async Buffer Save Triggered)", flush=True)
            else:
                d_save = time.time() - t_save_start
                print(f"Done in {d_save:.2f}s (Models Only)", flush=True)

        avg_adv = metrics['adv_losses'][-1] if metrics['adv_losses'] else 0.0
        avg_pol = metrics['pol_losses'][-1] if metrics['pol_losses'] else 0.0
        avg_len = np.mean(metrics['game_lengths'][-episodes_per_update:])
        
        if episodes_completed % 5000 == 0:
            print("  [Phase 6] Model Evaluation...", flush=True)
            t_eval_start = time.time()
            test_model(num_games=100, threads_per_worker=10, fast_pol_infer=fast_pol_infer)
            d_eval = time.time() - t_eval_start
            print(f"  -> Evaluation Finished in {d_eval:.2f}s", flush=True)
        else:
            print("  [Phase 6] Model Evaluation...           Skipped this round.", flush=True)
            
        elapsed = time.time() - start_time
        
        print("\n" + "="*60, flush=True)
        print(f"Episodes {episodes_completed}/{total_episodes} | Total Time: {elapsed:.1f}s | Current LR: {current_lr:.6f}", flush=True)
        print(f"  -> Win Rate: P1 ({metrics['p1_wins']}) vs P2 ({metrics['p2_wins']})", flush=True)
        print(f"  -> Avg Game Length: {avg_len:.1f} turns", flush=True)
        print(f"  -> Adv Buffer: {replay_adv_size} / {MAX_BUFFER_SIZE} | Pol Buffer: {replay_pol_size} / {MAX_BUFFER_SIZE}", flush=True)
        print(f"  -> Adv Loss: {avg_adv:.4f} | Pol Loss: {avg_pol:.4f}", flush=True)
        print("-" * 60, flush=True)
        metrics['p1_wins'], metrics['p2_wins'] = 0, 0

    print(f"\nTraining Complete! Final models secured.", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    train_self_play(total_episodes=20000000, episodes_per_update=1000, batch_size=4096, threads_per_worker=30)
