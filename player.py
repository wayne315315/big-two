import itertools
from helper import RANKS, SUITS, get_card_value, evaluate_play, is_valid_beat

# --- PLAYER INTERFACE ---
class Player:
    """Base interface for a Big Two player."""
    def __init__(self, name):
        self.name = name
        self.hand = []

    def receive_cards(self, cards):
        """Adds cards to the player's hand and sorts them."""
        self.hand.extend(cards)
        self.hand.sort(key=get_card_value)

    def remove_cards(self, cards_to_remove):
        """Removes played cards from the player's hand."""
        for card in cards_to_remove:
            self.hand.remove(card)

    def get_play(self, game_state):
        """
        Takes the current game state and returns a list of cards to play.
        An empty list [] represents a pass.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_play()")

# --- HUMAN PLAYER IMPLEMENTATION ---
class HumanPlayer(Player):
    def print_hand(self):
        """Formats the hand nicely with index numbers for the player to choose from."""
        display = [f"[{i}] {c[0]}-{c[1][0]}" for i, c in enumerate(self.hand)]
        print("  ".join(display))

    def get_play(self, game_state):
        print(f"\n--- {self.name}'s Turn ---")
        self.print_hand()
        
        while True:
            action = input("Enter card indices to play (comma-separated) or 'p' to pass: ").strip().lower()
            if action == 'p':
                return [] # Empty list implies passing
                
            try:
                indices = [int(i.strip()) for i in action.split(',')]
                selected_cards = [self.hand[i] for i in indices]
                return selected_cards
            except (ValueError, IndexError):
                print("Invalid input. Please enter valid numbers separated by commas.")

# --- BOT PLAYER IMPLEMENTATION ---
class BotPlayer(Player):
    def __init__(self, name="Bot"):
        super().__init__(name)

    def _get_legal_actions(self, game_state):
        """Generates all legally valid combinations we can play (Required for UI display)."""
        valid_actions = [[]] if game_state.get('table_eval') else []
        
        # 1. Singles
        for card in self.hand:
            eval_res = evaluate_play([card])
            if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                valid_actions.append([card])
                
        # 2. Pairs 
        rank_groups = {}
        for card in self.hand:
            rank_groups[card[0]] = rank_groups.get(card[0], []) + [card]
            
        for cards in rank_groups.values():
            if len(cards) >= 2:
                # Generate all possible 2-card combinations for this rank
                for pair in itertools.combinations(cards, 2):
                    pair_list = list(pair)
                    eval_res = evaluate_play(pair_list)
                    if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                        valid_actions.append(pair_list)
                        
        # 3. 5-Card Hands (Straights, Full Houses, Quads, Straight Flushes)
        if len(self.hand) >= 5:
            for combo in itertools.combinations(self.hand, 5):
                combo_list = list(combo)
                eval_res = evaluate_play(combo_list)
                if eval_res and is_valid_beat(eval_res, game_state.get('table_eval')):
                    valid_actions.append(combo_list)
                
        # First turn constraint
        if game_state.get('is_first_turn'):
            req_card = game_state['lowest_card']
            valid_actions = [a for a in valid_actions if req_card in a]
            
        return valid_actions

    def get_play(self, game_state):
        #print(f"\n--- {self.name}'s Turn (Bot) ---")
        
        # 1. First Turn Constraint
        if game_state['is_first_turn']:
            #print(f"{self.name} automatically plays the starting card.")
            return [game_state['lowest_card']]

        # 2. Analyze Hand for basic combinations
        rank_groups = {}
        for card in self.hand:
            rank_groups[card[0]] = rank_groups.get(card[0], []) + [card]
        
        pairs = [cards[:2] for cards in rank_groups.values() if len(cards) >= 2]

        # 3. Table Control (Free Play)
        if not game_state['table_eval']:
            if pairs:
                return pairs[0]
            return [self.hand[0]]

        # 4. Reacting to the Table
        table_type = game_state['table_eval'][0]
        
        if table_type == 'Single':
            for card in self.hand:
                if is_valid_beat(evaluate_play([card]), game_state['table_eval']):
                    return [card]
                    
        elif table_type == 'Pair':
            for pair in pairs:
                if is_valid_beat(evaluate_play(pair), game_state['table_eval']):
                    return pair
                    
        return []