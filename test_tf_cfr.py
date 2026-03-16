import random
import sys
import os

from helper import RANKS, SUITS, get_card_value, evaluate_play
from player import BotPlayer
from tf_deep_cfr_bot import TFDeepCFRBot

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def test_model(num_games=500, policy_path="tf_policy_net.weights.h5"):
    print("="*50)
    print(f"BENCHMARKING DEEP CFR POLICY VS STANDARD BOT ({num_games} GAMES)")
    print("="*50)
    
    # is_training=False forces exploration to 0.0 and utilizes the Policy Network
    cfr_bot = TFDeepCFRBot(name="TF Deep CFR Bot", policy_model_path=policy_path, is_training=False)
    standard_bot = BotPlayer("Standard Bot")
    
    cfr_wins = 0
    standard_wins = 0

    print("\nTesting in progress... (Gameplay outputs are suppressed)")
    
    for i in range(1, num_games + 1):
        deck = [(rank, suit) for rank in RANKS for suit in SUITS]
        random.shuffle(deck)
        piles = [deck[j * 13 : (j + 1) * 13] for j in range(4)]
        
        cfr_bot.hand = []
        standard_bot.hand = []
        cfr_bot.receive_cards(piles[0])
        standard_bot.receive_cards(piles[1])
        
        players = [cfr_bot, standard_bot]
        
        if get_card_value(cfr_bot.hand[0]) < get_card_value(standard_bot.hand[0]):
            current_idx = 0
            lowest_card = cfr_bot.hand[0]
        else:
            current_idx = 1
            lowest_card = standard_bot.hand[0]
            
        game_state = {
            'table_eval': None,
            'table_cards': [],
            'is_first_turn': True,
            'lowest_card': lowest_card,
            'dead_cards': [] 
        }
        
        last_player_idx = None
        
        with SuppressPrint():
            while True:
                current_player = players[current_idx]
                
                if last_player_idx == current_idx:
                    game_state['dead_cards'].extend(game_state['table_cards'])
                    game_state['table_eval'] = None
                    game_state['table_cards'] = []
                    
                selected_cards = current_player.get_play(game_state)
                
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
            cfr_wins += 1
        else:
            standard_wins += 1
            
        if i % 50 == 0 or i == num_games:
            progress = (i / num_games) * 100
            print(f"[{progress:>5.1f}%] Played {i} games...")

    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Games Played: {num_games}")
    print(f"CFR Bot Wins:       {cfr_wins} ({(cfr_wins/num_games)*100:.1f}%)")
    print(f"Standard Bot Wins:  {standard_wins} ({(standard_wins/num_games)*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    test_model(num_games=500)