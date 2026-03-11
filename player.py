from helper import RANKS, SUITS, get_card_value, evaluate_play, is_valid_beat

# --- PLAYER INTERFACE ---
class Player:
    """Base interface for a Big Two player."""
    def __init__(self, name):
        self.name = name
        self.hand = []

    def receive_cards(self, cards):
        self.hand.extend(cards)
        self.hand.sort(key=get_card_value)

    def remove_cards(self, cards_to_remove):
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

class BotPlayer(Player):
    def __init__(self, name="Bot"):
        super().__init__(name)

    def get_play(self, game_state):
        print(f"\n--- {self.name}'s Turn (Bot) ---")
        
        # 1. First Turn Constraint
        # The bot simply plays the required lowest card as a single to ensure validity.
        if game_state['is_first_turn']:
            print(f"{self.name} automatically plays the starting card.")
            return [game_state['lowest_card']]

        # 2. Analyze Hand for basic combinations
        # Group cards by rank to easily find pairs and triples.
        rank_groups = {}
        for card in self.hand:
            rank_groups[card[0]] = rank_groups.get(card[0], []) + [card]
        
        # Extract all available pairs and triples (sorted lowest to highest)
        pairs = [cards[:2] for cards in rank_groups.values() if len(cards) >= 2]
        triples = [cards[:3] for cards in rank_groups.values() if len(cards) >= 3]

        # 3. Table Control (Free Play)
        # If the bot won the last trick, it plays its lowest pair. 
        # If it has no pairs, it plays its lowest single card.
        if not game_state['table_eval']:
            if pairs:
                return pairs[0]
            return [self.hand[0]]

        # 4. Reacting to the Table
        table_type = game_state['table_eval'][0]
        
        if table_type == 'Single':
            # Find the lowest single card that legally beats the table
            for card in self.hand:
                if is_valid_beat(evaluate_play([card]), game_state['table_eval']):
                    return [card]
                    
        elif table_type == 'Pair':
            # Find the lowest pair that legally beats the table
            for pair in pairs:
                if is_valid_beat(evaluate_play(pair), game_state['table_eval']):
                    return pair
                    
        elif table_type == 'Triple':
            # Find the lowest triple that legally beats the table
            for triple in triples:
                if is_valid_beat(evaluate_play(triple), game_state['table_eval']):
                    return triple
                    
        # Note: This basic bot doesn't search for 5-card hands like Straights or Flushes.
        # If the human plays a 5-card hand, or if the bot cannot beat the table, it passes.
        return []
