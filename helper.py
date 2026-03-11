# --- CONSTANTS & RULES ---
RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUITS = ['Clubs', 'Diamonds', 'Hearts', 'Spades']

# Removed 'Flush' from the game. 
# Straight and Full House are independent and cannot beat each other unless a Bomb is used.
HAND_TYPES = {
    'Straight': 1,
    'FullHouse': 2,
    'Quads': 3,
    'StraightFlush': 4
}

# Real Big Two Straight Rules mapping: 
# (sorted_rank_indices) -> (straight_strength, index_of_key_card_in_sorted_hand)
STRAIGHTS = {
    (0, 1, 2, 11, 12): (0, 2), # A-2-3-4-5 (Smallest), Key Card is 5 (at index 2)
    (0, 1, 2, 3, 4): (1, 4),   # 3-4-5-6-7, Key Card is 7 (at index 4)
    (1, 2, 3, 4, 5): (2, 4),   # 4-5-6-7-8, Key Card is 8
    (2, 3, 4, 5, 6): (3, 4),   # 5-6-7-8-9, Key Card is 9
    (3, 4, 5, 6, 7): (4, 4),   # 6-7-8-9-10, Key Card is 10
    (4, 5, 6, 7, 8): (5, 4),   # 7-8-9-10-J, Key Card is J
    (5, 6, 7, 8, 9): (6, 4),   # 8-9-10-J-Q, Key Card is Q
    (6, 7, 8, 9, 10): (7, 4),  # 9-10-J-Q-K, Key Card is K
    (7, 8, 9, 10, 11): (8, 4), # 10-J-Q-K-A, Key Card is A
    (0, 1, 2, 3, 12): (9, 4),  # 2-3-4-5-6 (Largest), Key Card is 2 (at index 4)
}

# --- HELPER FUNCTIONS ---
def get_card_value(card):
    """Returns a tuple of (rank_index, suit_index) for easy comparison."""
    rank, suit = card
    return (RANKS.index(rank), SUITS.index(suit))

def evaluate_play(cards):
    """Evaluates a list of cards and returns (play_type, key_card_value, hand_type_rank)"""
    if not cards: return None
    
    # Sort the submitted hand from lowest card value to highest
    cards = sorted(cards, key=get_card_value)
    n = len(cards)
    
    if n == 1:
        return ('Single', get_card_value(cards[0]), 0)
    if n == 2 and cards[0][0] == cards[1][0]:
        return ('Pair', get_card_value(cards[1]), 0)
        
    # NOTE: "Triple" (3-of-a-kind) has been completely removed based on the updated rules.
        
    if n == 5:
        suits = [c[1] for c in cards]
        ranks_indices = tuple(sorted([RANKS.index(c[0]) for c in cards]))
        rank_counts = {r: ranks_indices.count(r) for r in set(ranks_indices)}
        
        is_flush = len(set(suits)) == 1
        is_straight = ranks_indices in STRAIGHTS
        
        # 1. Straight Flush (Bomb)
        if is_straight and is_flush:
            straight_strength, key_index = STRAIGHTS[ranks_indices]
            key_suit = SUITS.index(cards[key_index][1])
            return ('5-Card', (straight_strength, key_suit), HAND_TYPES['StraightFlush'])
            
        # 2. Quads / Four of a Kind (Bomb)
        if 4 in rank_counts.values():
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            key_card = max([c for c in cards if RANKS.index(c[0]) == quad_rank], key=get_card_value)
            return ('5-Card', get_card_value(key_card), HAND_TYPES['Quads'])
            
        # 3. Full House
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            key_card = max([c for c in cards if RANKS.index(c[0]) == trip_rank], key=get_card_value)
            return ('5-Card', get_card_value(key_card), HAND_TYPES['FullHouse'])
            
        # 4. Straight (Standard Flush is removed)
        if is_straight:
            straight_strength, key_index = STRAIGHTS[ranks_indices]
            key_suit = SUITS.index(cards[key_index][1])
            return ('5-Card', (straight_strength, key_suit), HAND_TYPES['Straight'])

    return None

def is_valid_beat(current_eval, previous_eval):
    """Determines if the current play legally beats the previous play."""
    if not previous_eval: 
        return True # Free play if there's no previous play on the table
    
    curr_type, curr_key, curr_hand_rank = current_eval
    prev_type, prev_key, prev_hand_rank = previous_eval
    
    # Check if the current or previous plays are "Bombs" (Quads or Straight Flush)
    is_curr_bomb = (curr_type == '5-Card' and curr_hand_rank in [HAND_TYPES['Quads'], HAND_TYPES['StraightFlush']])
    is_prev_bomb = (prev_type == '5-Card' and prev_hand_rank in [HAND_TYPES['Quads'], HAND_TYPES['StraightFlush']])
    
    # 1. Bomb Privilege: If the current play is a bomb and the table is NOT a bomb, 
    # it legally beats any single, pair, or standard 5-card hand.
    if is_curr_bomb and not is_prev_bomb:
        return True
        
    # 2. Quantity Constraint: Must match exact number of cards.
    if curr_type != prev_type: 
        return False
    
    # 3. 5-Card Strict Rules
    if curr_type == '5-Card':
        # If both players played bombs, Straight Flush beats Quads
        if is_curr_bomb and is_prev_bomb:
            if curr_hand_rank > prev_hand_rank: return True
            if curr_hand_rank < prev_hand_rank: return False
        # If neither are bombs, they MUST be the exact same hand type 
        # (e.g., Straight vs Straight, or FullHouse vs FullHouse). They cannot beat each other.
        else:
            if curr_hand_rank != prev_hand_rank:
                return False
        
    # 4. Key Card Comparison: If the hand types are exactly the same, compare their key cards.
    return curr_key > prev_key
