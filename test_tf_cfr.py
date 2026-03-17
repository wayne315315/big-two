import os
import time
import random
import multiprocessing as mp
import multiprocessing.connection 
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

from helper import RANKS, SUITS, get_card_value, evaluate_play
from player import BotPlayer
from tf_deep_cfr_bot import TFDeepCFRBot, create_policy_network

# ==============================================================================
# THE GPU INFERENCE SERVER (STATIC XLA BUFFERING)
# ==============================================================================
# BOTTLECK FIX: Test server now accepts the pre-compiled graph to save 10+ seconds
def gpu_inference_server(conns, pol_net=None, fast_pol_infer=None):
    FIXED_BATCH = 16384
    
    if fast_pol_infer is None:
        @tf.function(input_signature=[tf.TensorSpec(shape=(FIXED_BATCH, 37, 55), dtype=tf.float16)], jit_compile=True)
        def fast_pol_infer_internal(batch):
            return pol_net(batch, training=False)
        fast_pol_infer = fast_pol_infer_internal

        print("Compiling XLA Transformer Kernel (This takes a few seconds)...")
        _ = fast_pol_infer(tf.zeros((FIXED_BATCH, 37, 55), dtype=tf.float16))
        print("XLA Compilation Complete! Benchmark Engine Armed.")

    active_conns = list(conns)
    
    MIN_BATCH_SIZE = 4096
    MAX_WAIT_TIME = 0.05

    pol_buffer = np.zeros((FIXED_BATCH, 37, 55), dtype=np.float16)
    pol_pipes, pol_lens = [], []
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
                    if is_policy:  
                        length = len(inputs)
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
            except EOFError:
                if conn in active_conns:
                    active_conns.remove(conn)
                    
        time_waiting = time.time() - last_fire_time
        
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

# ==============================================================================
# INTRA-PROCESS THREAD LOGIC
# ==============================================================================
def _thread_test_games(num_games, conn):
    cfr_bot = TFDeepCFRBot("CFR Bot", pipe=conn, is_training=False)
    standard_bot = BotPlayer("Standard Bot")
    
    cfr_wins, standard_wins = 0, 0
    
    for _ in range(num_games):
        deck = [(rank, suit) for rank in RANKS for suit in SUITS]
        random.shuffle(deck)
        piles = [deck[j * 13 : (j + 1) * 13] for j in range(4)]
        
        cfr_bot.hand, standard_bot.hand = [], []
        cfr_bot.receive_cards(piles[0])
        standard_bot.receive_cards(piles[1])
        players = [cfr_bot, standard_bot]
        
        current_idx = 0 if get_card_value(cfr_bot.hand[0]) < get_card_value(standard_bot.hand[0]) else 1
        lowest_card = cfr_bot.hand[0] if current_idx == 0 else standard_bot.hand[0]
            
        game_state = { 
            'table_eval': None, 
            'table_cards': [], 
            'is_first_turn': True, 
            'lowest_card': lowest_card, 
            'dead_cards': [], 
            'history': [] 
        }
        last_player_idx = None
        
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
            
            if not selected_cards:
                if not game_state['table_eval']: selected_cards = [current_player.hand[0]]
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
                if current_idx == 0: cfr_wins += 1
                else: standard_wins += 1
                game_state['dead_cards'].extend(game_state['table_cards'])
                break
            current_idx = 1 - current_idx
            
    return cfr_wins, standard_wins

# ==============================================================================
# CPU MULTIPROCESS MANAGERS
# ==============================================================================
def distributed_test_worker(num_games, conns, result_queue):
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
                futures.append(executor.submit(_thread_test_games, thread_tasks[i], conns[i]))
        
        for future in futures:
            results.append(future.result())

    total_cfr_wins, total_std_wins = 0, 0
    for cw, sw in results:
        total_cfr_wins += cw
        total_std_wins += sw

    for conn in conns:
        conn.send("DONE") 

    result_queue.put((total_cfr_wins, total_std_wins))

# ==============================================================================
# MAIN TEST ORCHESTRATOR
# ==============================================================================
# Modified signature to accept pre-compiled functions
def test_model(num_games=1000, policy_path="tf_policy_model.keras", num_workers=None, threads_per_worker=10, fast_pol_infer=None):
    print("="*60)
    print(f"DISTRIBUTED FLOAT16 GPU BENCHMARK ({num_games} GAMES)")
    print("="*60)
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    # Prevent over-spawning if num_games is very small
    if num_games < num_workers * threads_per_worker:
        num_workers = max(1, num_games // threads_per_worker)
        if num_workers == 0:
            num_workers = 1
            threads_per_worker = num_games
        
    total_threads = num_workers * threads_per_worker
    print(f"Spawning {num_workers} processes x {threads_per_worker} threads ({total_threads} virtual actors) utilizing 1 GPU...")
    start_time = time.time()
    
    if fast_pol_infer is None:
        try:
            shared_pol_net = tf.keras.models.load_model(policy_path, compile=False)
            print(f"Successfully loaded full model '{policy_path}'")
        except:
            print(f"Warning: Could not load '{policy_path}'. Using random weights.")
            shared_pol_net = create_policy_network()
    else:
        shared_pol_net = None # Skip loading model if function is already compiled
        
    ctx = mp.get_context('spawn')
    
    base_games = num_games // num_workers
    remainder = num_games % num_workers
    test_tasks = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]
    
    pipes = [ctx.Pipe() for _ in range(total_threads)]
    parent_conns = [p[0] for p in pipes]
    child_conns = [p[1] for p in pipes]
    
    child_conn_chunks = [child_conns[i * threads_per_worker : (i + 1) * threads_per_worker] for i in range(num_workers)]
    
    result_queue = ctx.Queue()

    processes = []
    for i, task_count in enumerate(test_tasks):
        if task_count > 0:
            p = ctx.Process(target=distributed_test_worker, args=(task_count, child_conn_chunks[i], result_queue))
            p.start()
            processes.append(p)

    gpu_inference_server(parent_conns, shared_pol_net, fast_pol_infer)

    total_cfr_wins, total_std_wins = 0, 0
    for _ in range(len(processes)):
        cw, sw = result_queue.get()
        total_cfr_wins += cw
        total_std_wins += sw

    for p in processes:
        p.join()
        
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"FINAL BENCHMARK RESULTS (Completed in {elapsed:.1f}s)")
    print("="*60)
    print(f"Total Games Played: {num_games}")
    print(f"CFR Bot Wins:       {total_cfr_wins} ({(total_cfr_wins/num_games)*100:.1f}%)")
    print(f"Standard Bot Wins:  {total_std_wins} ({(total_std_wins/num_games)*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    mp.freeze_support()
    test_model(num_games=1000, threads_per_worker=30)
