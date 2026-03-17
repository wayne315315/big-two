from flask import Flask, render_template, jsonify, request
import random

# Import your existing game logic
from helper import RANKS, SUITS, get_card_value, evaluate_play, is_valid_beat
from player import Player
from tf_deep_cfr_bot import TFDeepCFRBot

app = Flask(__name__)

# Global variables to hold the game state in memory
game_instance = {}

def init_game():
    """Initializes the deck, deals cards, and determines the starting player."""
    global game_instance
    
    deck = [(rank, suit) for rank in RANKS for suit in SUITS]
    random.shuffle(deck)
    piles = [deck[i * 13 : (i + 1) * 13] for i in range(4)]
    
    human = Player("Player 1 (You)")
    bot = TFDeepCFRBot("CFR Bot")
    
    human.receive_cards(piles[0])
    bot.receive_cards(piles[1])
    
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
            'lowest_card': lowest_card,
            'dead_cards': [],
            'history': [] # INJECT HISTORY
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset():
    init_game()
    return get_state()

@app.route('/api/state', methods=['GET'])
def get_state():
    if not game_instance:
        init_game()
        
    gs = game_instance['game_state_dict']
    bot = game_instance['bot']
    human = game_instance['human']
    
    bot_legal_actions = []
    is_game_over = len(human.hand) == 0 or len(bot.hand) == 0
    
    if not is_game_over:
        # Pre-feed the required variables for the Bot's inference
        gs['my_idx'] = 1
        gs['my_hand_size'] = len(bot.hand)
        gs['opp_hand_size'] = len(human.hand)
        bot_legal_actions = bot._get_legal_actions(gs)
        
    return jsonify({
        'human_hand': human.hand,
        'bot_hand': bot.hand,  
        'bot_card_count': len(bot.hand),
        'table_cards': gs['table_cards'],
        'dead_cards': gs['dead_cards'], 
        'bot_legal_actions': bot_legal_actions,  
        'current_turn': game_instance['current_idx'], 
        'message': game_instance['message'],
        'is_game_over': is_game_over
    })

@app.route('/api/play', methods=['POST'])
def play_cards():
    data = request.json
    selected_cards = [tuple(c) for c in data.get('cards', [])]
    
    if not selected_cards:
        return jsonify({'error': 'Please select cards to play.'}), 400

    human = game_instance['human']
    gs = game_instance['game_state_dict']

    if gs['is_first_turn'] and gs['lowest_card'] not in selected_cards:
        req_card = gs['lowest_card']
        return jsonify({'error': f"You must include the {req_card[0]} of {req_card[1]} on the first turn."}), 400

    curr_eval = evaluate_play(selected_cards)
    if not curr_eval:
        return jsonify({'error': 'Not a recognized Big Two hand.'}), 400

    if not is_valid_beat(curr_eval, gs['table_eval']):
        return jsonify({'error': 'Your play is not high enough to beat the table, or does not match the card type.'}), 400

    # Apply valid human play
    human.remove_cards(selected_cards)
    
    gs['history'].append((0, selected_cards)) # Human played cards
    
    if gs['table_cards']:
        gs['dead_cards'].extend(gs['table_cards'])
        
    gs['table_eval'] = curr_eval
    gs['table_cards'] = selected_cards
    gs['is_first_turn'] = False
    game_instance['last_player_idx'] = 0
    
    game_instance['message'] = f"You played: {', '.join([f'{c[0]} of {c[1]}' for c in selected_cards])}"
    
    if not human.hand:
        game_instance['message'] = "YOU WIN THE GAME!"
    else:
        game_instance['current_idx'] = 1
        check_table_control()

    return get_state()

@app.route('/api/pass', methods=['POST'])
def pass_turn():
    gs = game_instance['game_state_dict']
    if not gs['table_eval']:
        return jsonify({'error': 'You cannot pass! You control the table.'}), 400
        
    gs['history'].append((0, [])) # Human passed
    
    game_instance['message'] = "You passed."
    game_instance['current_idx'] = 1
    check_table_control()
    return get_state()

@app.route('/api/bot_turn', methods=['POST'])
def bot_turn():
    bot = game_instance['bot']
    human = game_instance['human']
    gs = game_instance['game_state_dict']
    
    # Pre-feed metrics
    gs['my_idx'] = 1
    gs['my_hand_size'] = len(bot.hand)
    gs['opp_hand_size'] = len(human.hand)
    
    selected_cards = bot.get_play(gs)
    
    if not selected_cards:
        if not gs['table_eval']:
            selected_cards = [bot.hand[0]] # Failsafe
        else:
            gs['history'].append((1, [])) # Bot passed
            game_instance['message'] = "Bot passed."
            game_instance['current_idx'] = 0
            check_table_control()
            return get_state()
            
    curr_eval = evaluate_play(selected_cards)
    bot.remove_cards(selected_cards)
    
    gs['history'].append((1, selected_cards)) # Bot played cards
    
    if gs['table_cards']:
        gs['dead_cards'].extend(gs['table_cards'])
        
    gs['table_eval'] = curr_eval
    gs['table_cards'] = selected_cards
    gs['is_first_turn'] = False
    game_instance['last_player_idx'] = 1
    
    game_instance['message'] = f"Bot played: {', '.join([f'{c[0]} of {c[1]}' for c in selected_cards])}"
    
    if not bot.hand:
        game_instance['message'] = "BOT WINS THE GAME!"
    else:
        game_instance['current_idx'] = 0
        check_table_control()
        
    return get_state()

def check_table_control():
    gs = game_instance['game_state_dict']
    if game_instance['last_player_idx'] == game_instance['current_idx']:
        if gs['table_cards']:
            gs['dead_cards'].extend(gs['table_cards'])
            
        gs['table_eval'] = None
        gs['table_cards'] = []

if __name__ == '__main__':
    app.run(debug=True)
