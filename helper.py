# --- CONSTANTS & RULES ---
RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUITS = ['Clubs', 'Diamonds', 'Hearts', 'Spades']

HAND_TYPES = {
    'Straight': 1,
    'Flush': 2,
    'FullHouse': 3,
    'Quads': 4,
    'StraightFlush': 5
}

# --- HELPER FUNCTIONS ---
def get_card_value(card):
    rank, suit = card
    return (RANKS.index(rank), SUITS.index(suit))

def evaluate_play(cards):
    """Evaluates a list of cards and returns (play_type, key_card_value, hand_type_rank)"""
    if not cards: return None
    cards = sorted(cards, key=get_card_value)
    n = len(cards)
    
    if n == 1:
        return ('Single', get_card_value(cards[0]), 0)
    if n == 2 and cards[0][0] == cards[1][0]:
        return ('Pair', get_card_value(cards[1]), 0)
    if n == 3 and cards[0][0] == cards[1][0] == cards[2][0]:
        return ('Triple', get_card_value(cards[2]), 0)
        
    if n == 5:
        suits = [c[1] for c in cards]
        ranks = [RANKS.index(c[0]) for c in cards]
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        
        is_flush = len(set(suits)) == 1
        is_straight = all(ranks[i] == ranks[i-1] + 1 for i in range(1, 5))
        
        if is_straight and is_flush:
            return ('5-Card', get_card_value(cards[-1]), HAND_TYPES['StraightFlush'])
        if 4 in rank_counts.values():
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            key_card = max([c for c in cards if RANKS.index(c[0]) == quad_rank], key=get_card_value)
            return ('5-Card', get_card_value(key_card), HAND_TYPES['Quads'])
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            key_card = max([c for c in cards if RANKS.index(c[0]) == trip_rank], key=get_card_value)
            return ('5-Card', get_card_value(key_card), HAND_TYPES['FullHouse'])
        if is_flush:
            return ('5-Card', get_card_value(cards[-1]), HAND_TYPES['Flush'])
        if is_straight:
            return ('5-Card', get_card_value(cards[-1]), HAND_TYPES['Straight'])

    return None

def is_valid_beat(current_eval, previous_eval):
    if not previous_eval: return True
    curr_type, curr_key, curr_hand_rank = current_eval
    prev_type, prev_key, prev_hand_rank = previous_eval
    
    if curr_type != prev_type: return False
    if curr_type == '5-Card':
        if curr_hand_rank > prev_hand_rank: return True
        if curr_hand_rank < prev_hand_rank: return False
    return curr_key > prev_key
