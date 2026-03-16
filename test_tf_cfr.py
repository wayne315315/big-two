import os
import time
import random
import sys
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tensorflow as tf

from helper import RANKS, SUITS, get_card_value, evaluate_play
from player import BotPlayer
from tf_deep_cfr_bot import TFDeepCFRBot, create_policy_network

# ==============================================================================
# THE GPU INFERENCE SERVER (TESTING MODE)
# ==============================================================================
def gpu_inference_server(conns, pol_net):
    """Continuously polls pipes, batches data, and evaluates on the GPU."""
    active_conns = list(conns)
    while active_conns:
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
                        # Testing mode only ever requests the Policy Network
                        if is_policy:  
                            pol_reqs.append(inputs)
                            pol_pipes.append(conn)
                            pol_lens.append(len(inputs))
                except EOFError:
                    active_conns.remove(conn)
        
        if not processed_any:
            time.sleep(0.001)
            continue

        # GPU Dynamic Batching
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
def _thread_test_games(num_games, conn):
    """The isolated game loop executed by individual threads."""
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
            
        game_state = { 'table_eval': None, 'table_cards': [], 'is_first_turn': True, 'lowest_card': lowest_card, 'dead_cards': [] }
        last_player_idx = None
        
        while True:
            current_player = players[current_idx]
            if last_player_idx == current_idx:
                game_state['dead_cards'].extend(game_state['table_cards'])
                game_state['table_eval'] = None
                game_state['table_cards'] = []
                
            selected_cards = current_player.get_play(game_state)
            
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
    """Process Manager: Spawns threads to overlap Python processing with GPU Wait times."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Blind the actor to GPU
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
                futures.append(executor.submit(_thread_test_games, thread_tasks[i], conns[i]))
        
        for future in futures:
            results.append(future.result())

    total_cfr_wins, total_std_wins = 0, 0
    for cw, sw in results:
        total_cfr_wins += cw
        total_std_wins += sw

    for conn in conns:
        conn.send("DONE") # Tell the server this worker is finished

    result_queue.put((total_cfr_wins, total_std_wins))

# ==============================================================================
# MAIN TEST ORCHESTRATOR
# ==============================================================================
def test_model(num_games=1000, policy_path="tf_policy_net.weights.h5", num_workers=None, threads_per_worker=4):
    print("="*60)
    print(f"DISTRIBUTED MULTI-THREADED GPU BENCHMARK ({num_games} GAMES)")
    print("="*60)
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    total_threads = num_workers * threads_per_worker
    print(f"Spawning {num_workers} processes x {threads_per_worker} threads ({total_threads} virtual actors) utilizing 1 GPU...")
    start_time = time.time()
    
    # Initialize GPU Policy Network (Testing does not require the Advantage Network)
    shared_pol_net = create_policy_network()
    try:
        shared_pol_net.load_weights(policy_path)
        print(f"Successfully loaded '{policy_path}'")
    except:
        print(f"Warning: Could not load '{policy_path}'. Using random weights.")
        
    ctx = mp.get_context('spawn')
    
    # Distribute games evenly across processes
    base_games = num_games // num_workers
    remainder = num_games % num_workers
    test_tasks = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]
    
    # Create a massive bank of pipes (1 for every thread)
    pipes = [ctx.Pipe() for _ in range(total_threads)]
    parent_conns = [p[0] for p in pipes]
    child_conns = [p[1] for p in pipes]
    
    # Group the child pipes into chunks for the processes
    child_conn_chunks = [child_conns[i * threads_per_worker : (i + 1) * threads_per_worker] for i in range(num_workers)]
    
    result_queue = ctx.Queue()

    # Start Worker Processes
    processes = []
    for i, task_count in enumerate(test_tasks):
        if task_count > 0:
            p = ctx.Process(target=distributed_test_worker, args=(task_count, child_conn_chunks[i], result_queue))
            p.start()
            processes.append(p)

    # Run GPU Inference Server (Blocks until all workers send "DONE")
    gpu_inference_server(parent_conns, shared_pol_net)

    # Harvest Results
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
    # Set to run a 1,000-game test. You can easily bump this to 5,000 or 10,000.
    test_model(num_games=1000, threads_per_worker=100)