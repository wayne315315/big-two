import random

from helper import RANKS, SUITS, get_card_value, evaluate_play, is_valid_beat
from player import HumanPlayer, BotPlayer

# --- GAME ENGINE ---
def play_game():
    # 1. Setup Deck & Players
    deck = [(rank, suit) for rank in RANKS for suit in SUITS]
    random.shuffle(deck)
    # Distribute into 4 equal piles of 13 cards each
    piles = [deck[i * 13 : (i + 1) * 13] for i in range(4)]
    
    player1 = HumanPlayer("Player 1")
    player2 = BotPlayer("Big Two Bot")
    
    # Assign to 2 players and discard the remaining 2 piles
    player1.receive_cards(piles[0])
    player2.receive_cards(piles[1])
    
    players = [player1, player2]
    
    # Determine starting player based on absolute lowest card
    p1_lowest = player1.hand[0]
    p2_lowest = player2.hand[0]
    
    if get_card_value(p1_lowest) < get_card_value(p2_lowest):
        current_idx = 0
        lowest_card = p1_lowest
    else:
        current_idx = 1
        lowest_card = p2_lowest
        
    print(f"\nGame Start! {players[current_idx].name} goes first because they hold the {lowest_card[0]} of {lowest_card[1]}.")
    
    # Game State Variables
    game_state = {
        'table_eval': None,
        'table_cards': [],
        'is_first_turn': True,
        'lowest_card': lowest_card
    }
    
    last_player_idx = None
    
    # 2. Main Game Loop
    while True:
        print("\n" + "="*40)
        current_player = players[current_idx]
        
        # If the turn comes back to the last person who played, they win the trick and clear the table
        if last_player_idx == current_idx:
            print(f"{players[1 - current_idx].name} passed. {current_player.name} takes the table!")
            game_state['table_eval'] = None
            game_state['table_cards'] = []
            
        if game_state['table_cards']:
            display_table = [f"{c[0]}-{c[1][0]}" for c in game_state['table_cards']]
            print(f"ON TABLE: {' '.join(display_table)}")
        else:
            print("ON TABLE: [Empty - Play any valid combination]")
            
        # Keep asking the current player for a move until they provide a legally valid one
        while True:
            selected_cards = current_player.get_play(game_state)
            
            # Handle Passing
            if not selected_cards:
                if not game_state['table_eval']:
                    print("Invalid play: You cannot pass! You control the table. You must play something.")
                    continue
                print(f"{current_player.name} passes.")
                break
                
            # Handle First Turn Constraint
            if game_state['is_first_turn'] and game_state['lowest_card'] not in selected_cards:
                print(f"Invalid play: You must include the starting card ({game_state['lowest_card'][0]} of {game_state['lowest_card'][1]}) on the first turn.")
                continue
                
            # Handle Evaluation & Beating the Table
            curr_eval = evaluate_play(selected_cards)
            if not curr_eval:
                print("Invalid combination. Not a recognized Big Two hand.")
                continue
                
            if not is_valid_beat(curr_eval, game_state['table_eval']):
                print("Invalid play: Your play is not high enough to beat the table, or does not match the card quantity/type.")
                continue
                
            # Play is valid! Apply to game state.
            print(f"{current_player.name} plays: {', '.join([f'{c[0]}-{c[1][0]}' for c in selected_cards])}")
            current_player.remove_cards(selected_cards)
            
            game_state['table_eval'] = curr_eval
            game_state['table_cards'] = selected_cards
            game_state['is_first_turn'] = False
            last_player_idx = current_idx
            break # Break out of validation loop
            
        # Check Win Condition
        if not current_player.hand:
            print("\n" + "*"*40)
            print(f"{current_player.name.upper()} WINS THE GAME!")
            print("*"*40)
            break
            
        # Switch turn (toggles between 0 and 1)
        current_idx = 1 - current_idx

if __name__ == "__main__":
    play_game()
