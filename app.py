from flask import Flask, render_template, jsonify, request
import random

# Import your existing game logic
from helper import RANKS, SUITS, get_card_value, evaluate_play, is_valid_beat
from player import Player, BotPlayer

app = Flask(__name__)

# Global variables to hold the game state in memory
game_instance = {}

def init_game():
    """Initializes the deck, deals cards, and determines the starting player."""
    global game_instance
    
    deck = [(rank, suit) for rank in RANKS for suit in SUITS]
    random.shuffle(deck)
    piles = [deck[i * 13 : (i + 1) * 13] for i in range(4)]
    
    # We use the base Player class for the human so it doesn't trigger the terminal input()
    human = Player("Player 1 (You)")
    bot = BotPlayer("Big Two Bot")
    
    human.receive_cards(piles[0])
    bot.receive_cards(piles[1])
    
    # Determine the absolute lowest card
    p1_lowest = human.hand[0]
    p2_lowest = bot.hand[0]
    
    if get_card_value(p1_lowest) < get_card_value(p2_lowest):
        current_idx = 0
        lowest_card = p1_lowest
    else:
        current_idx = 1
        lowest_card = p2_lowest

    game_instance = {
        'human': human,
        'bot': bot,
        'current_idx': current_idx,
        'last_player_idx': None,
        'message': f"Game Start! {'You go' if current_idx == 0 else 'Bot goes'} first.",
        'game_state_dict': {
            'table_eval': None,
            'table_cards': [],
            'is_first_turn': True,
            'lowest_card': lowest_card
        }
    }

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset():
    """Resets and starts a new game."""
    init_game()
    return get_state()

@app.route('/api/state', methods=['GET'])
def get_state():
    """Returns the current game state to the frontend UI."""
    if not game_instance:
        init_game()
        
    gs = game_instance['game_state_dict']
    return jsonify({
        'human_hand': game_instance['human'].hand,
        'bot_card_count': len(game_instance['bot'].hand),
        'table_cards': gs['table_cards'],
        'current_turn': game_instance['current_idx'], # 0 for Human, 1 for Bot
        'message': game_instance['message'],
        'is_game_over': len(game_instance['human'].hand) == 0 or len(game_instance['bot'].hand) == 0
    })

@app.route('/api/play', methods=['POST'])
def play_cards():
    """Handles the human player submitting cards."""
    data = request.json
    # Convert JSON lists back to tuples so they match our Python game logic
    selected_cards = [tuple(c) for c in data.get('cards', [])]
    
    if not selected_cards:
        return jsonify({'error': 'Please select cards to play.'}), 400

    human = game_instance['human']
    gs = game_instance['game_state_dict']

    # 1. Enforce First Turn Rule
    if gs['is_first_turn'] and gs['lowest_card'] not in selected_cards:
        req_card = gs['lowest_card']
        return jsonify({'error': f"You must include the {req_card[0]} of {req_card[1]} on the first turn."}), 400

    # 2. Evaluate Play
    curr_eval = evaluate_play(selected_cards)
    if not curr_eval:
        return jsonify({'error': 'Not a recognized Big Two hand.'}), 400

    # 3. Check against Table State
    if not is_valid_beat(curr_eval, gs['table_eval']):
        return jsonify({'error': 'Your play is not high enough to beat the table, or does not match the card type.'}), 400

    # Apply valid play
    human.remove_cards(selected_cards)
    gs['table_eval'] = curr_eval
    gs['table_cards'] = selected_cards
    gs['is_first_turn'] = False
    game_instance['last_player_idx'] = 0
    
    game_instance['message'] = f"You played: {', '.join([f'{c[0]} of {c[1]}' for c in selected_cards])}"
    
    # Check win
    if not human.hand:
        game_instance['message'] = "YOU WIN THE GAME!"
    else:
        game_instance['current_idx'] = 1 # Switch to Bot's turn
        check_table_control()

    return get_state()

@app.route('/api/pass', methods=['POST'])
def pass_turn():
    """Handles the human passing their turn."""
    gs = game_instance['game_state_dict']
    if not gs['table_eval']:
        return jsonify({'error': 'You cannot pass! You control the table.'}), 400
        
    game_instance['message'] = "You passed."
    game_instance['current_idx'] = 1
    check_table_control()
    return get_state()

@app.route('/api/bot_turn', methods=['POST'])
def bot_turn():
    """Triggers the bot's logic to make a move."""
    bot = game_instance['bot']
    gs = game_instance['game_state_dict']
    
    selected_cards = bot.get_play(gs)
    
    if not selected_cards:
        if not gs['table_eval']:
            selected_cards = [bot.hand[0]] # Failsafe
        else:
            game_instance['message'] = "Bot passed."
            game_instance['current_idx'] = 0
            check_table_control()
            return get_state()
            
    curr_eval = evaluate_play(selected_cards)
    bot.remove_cards(selected_cards)
    
    gs['table_eval'] = curr_eval
    gs['table_cards'] = selected_cards
    gs['is_first_turn'] = False
    game_instance['last_player_idx'] = 1
    
    game_instance['message'] = f"Bot played: {', '.join([f'{c[0]} of {c[1]}' for c in selected_cards])}"
    
    # Check win
    if not bot.hand:
        game_instance['message'] = "BOT WINS THE GAME!"
    else:
        game_instance['current_idx'] = 0
        check_table_control()
        
    return get_state()

def check_table_control():
    """Checks if a player won the trick (the other passed)."""
    if game_instance['last_player_idx'] == game_instance['current_idx']:
        game_instance['message'] += " The table is yours!"
        game_instance['game_state_dict']['table_eval'] = None
        game_instance['game_state_dict']['table_cards'] = []

if __name__ == '__main__':
    app.run(debug=True)