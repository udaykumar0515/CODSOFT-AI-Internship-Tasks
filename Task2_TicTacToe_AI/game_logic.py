"""
Game Logic and User Interface for Tic-Tac-Toe
CODSOFT AI Internship - Task 2
"""

from minimax import TicTacToeAI
import sys


class TicTacToeGame:
    def __init__(self, use_alpha_beta: bool = True):
        self.game = TicTacToeAI(use_alpha_beta)
        self.use_alpha_beta = use_alpha_beta
        
    def display_welcome_message(self) -> None:
        """Display welcome message and instructions"""
        print("=" * 60)
        print("🎮 TIC-TAC-TOE AI - CODSOFT AI INTERNSHIP")
        print("=" * 60)
        print("\nWelcome to Tic-Tac-Toe with Unbeatable AI!")
        print(f"Algorithm: Minimax {'with Alpha-Beta Pruning' if self.use_alpha_beta else 'without Alpha-Beta Pruning'}")
        print("\nGame Rules:")
        print("- You are X, AI is O")
        print("- Enter position numbers 1-9 as shown below:")
        self.show_position_guide()
        print("- AI uses Minimax algorithm for optimal play")
        print("- Try to beat the AI (it's impossible!)")
        print("- Type 'quit' to exit the game")
        print("\n" + "=" * 60)
    
    def show_position_guide(self) -> None:
        """Show position numbering guide"""
        guide_board = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        print(f" {guide_board[0]} | {guide_board[1]} | {guide_board[2]} ")
        print("---|---|---")
        print(f" {guide_board[3]} | {guide_board[4]} | {guide_board[5]} ")
        print("---|---|---")
        print(f" {guide_board[6]} | {guide_board[7]} | {guide_board[8]} ")
    
    def get_user_input(self) -> int:
        """Get and validate user input"""
        while True:
            try:
                user_input = input("Your turn (X). Enter position (1-9) or 'quit': ").strip().lower()
                
                if user_input == 'quit':
                    return -1
                
                position = int(user_input)
                
                if 1 <= position <= 9:
                    if self.game.make_human_move(position - 1):
                        return position
                    else:
                        print("Position already taken! Choose another position.")
                else:
                    print("Invalid position! Enter a number between 1 and 9.")
                    
            except ValueError:
                print("Invalid input! Enter a number between 1 and 9, or 'quit'.")
            except KeyboardInterrupt:
                print("\n\nGame interrupted. Goodbye!")
                sys.exit(0)
    
    def display_game_result(self, winner: str) -> None:
        """Display the game result"""
        print("\n" + "=" * 40)
        print("🎯 GAME OVER")
        print("=" * 40)
        
        if winner == 'X':
            print("🎉 Congratulations! You won!")
            print("(This should be impossible with perfect AI play)")
        elif winner == 'O':
            print("🤖 AI wins! Better luck next time!")
        else:
            print("🤝 It's a draw! Well played!")
        
        print(f"Nodes explored by AI: {self.game.ai.nodes_explored}")
        print("=" * 40)
    
    def ask_play_again(self) -> bool:
        """Ask if user wants to play again"""
        while True:
            try:
                choice = input("\nPlay again? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' or 'n'.")
            except KeyboardInterrupt:
                print("\n\nThanks for playing! Goodbye!")
                return False
    
    def play_game(self) -> None:
        """Main game loop"""
        self.display_welcome_message()
        
        while True:
            # Reset game
            self.game.reset_game()
            
            print("\n🎮 New Game Started!")
            print("You are X, AI is O")
            self.game.print_board()
            
            # Game loop
            while True:
                # Human move
                position = self.get_user_input()
                if position == -1:  # User wants to quit
                    print("\nThanks for playing! Goodbye!")
                    return
                
                print(f"You played position {position}")
                self.game.print_board()
                
                # Check if game is over after human move
                game_over, winner = self.game.is_game_over()
                if game_over:
                    self.display_game_result(winner)
                    break
                
                # AI move
                print("🤖 AI is thinking...")
                self.game.make_ai_move()
                self.game.print_board()
                
                # Check if game is over after AI move
                game_over, winner = self.game.is_game_over()
                if game_over:
                    self.display_game_result(winner)
                    break
            
            # Ask to play again
            if not self.ask_play_again():
                print("\nThanks for playing Tic-Tac-Toe AI! Goodbye!")
                break


def main():
    """Main function"""
    print("Choose AI algorithm:")
    print("1. Minimax with Alpha-Beta Pruning (faster)")
    print("2. Minimax without Alpha-Beta Pruning (slower, explores more nodes)")
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                use_alpha_beta = True
                break
            elif choice == '2':
                use_alpha_beta = False
                break
            else:
                print("Invalid choice! Enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            return
    
    game = TicTacToeGame(use_alpha_beta)
    game.play_game()


if __name__ == "__main__":
    main()
