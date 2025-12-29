"""
Minimax Algorithm with Alpha-Beta Pruning for Tic-Tac-Toe
CODSOFT AI Internship - Task 2
"""

import math
from typing import List, Tuple, Optional


class MinimaxAI:
    def __init__(self, use_alpha_beta: bool = True):
        self.use_alpha_beta = use_alpha_beta
        self.nodes_explored = 0
        
    def minimax(self, board: List[str], depth: int, is_maximizing: bool, 
                alpha: float = -math.inf, beta: float = math.inf) -> Tuple[int, Optional[int]]:
        """
        Minimax algorithm with optional alpha-beta pruning
        
        Returns:
            Tuple of (score, best_move)
            score: 1 (AI wins), 0 (draw), -1 (human wins)
            best_move: best position index (0-8) or None
        """
        self.nodes_explored += 1
        
        # Check terminal states
        winner = self.check_winner(board)
        if winner == 'O':  # AI wins
            return 1, None
        elif winner == 'X':  # Human wins
            return -1, None
        elif self.is_board_full(board):  # Draw
            return 0, None
        
        if is_maximizing:  # AI's turn (O)
            max_eval = -math.inf
            best_move = None
            
            for move in self.get_available_moves(board):
                board[move] = 'O'
                eval_score, _ = self.minimax(board, depth + 1, False, alpha, beta)
                board[move] = ' '  # Undo move
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
            
            return max_eval, best_move
        else:  # Human's turn (X)
            min_eval = math.inf
            best_move = None
            
            for move in self.get_available_moves(board):
                board[move] = 'X'
                eval_score, _ = self.minimax(board, depth + 1, True, alpha, beta)
                board[move] = ' '  # Undo move
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
            
            return min_eval, best_move
    
    def get_best_move(self, board: List[str]) -> int:
        """Get the best move for AI"""
        self.nodes_explored = 0
        _, best_move = self.minimax(board, 0, True)
        return best_move
    
    def get_available_moves(self, board: List[str]) -> List[int]:
        """Get list of available moves"""
        return [i for i, cell in enumerate(board) if cell == ' ']
    
    def is_board_full(self, board: List[str]) -> bool:
        """Check if board is full"""
        return ' ' not in board
    
    def check_winner(self, board: List[str]) -> Optional[str]:
        """Check if there's a winner"""
        # Winning combinations
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for pattern in win_patterns:
            a, b, c = pattern
            if board[a] == board[b] == board[c] != ' ':
                return board[a]
        
        return None
    
    def evaluate_board(self, board: List[str]) -> int:
        """Simple board evaluation (not used in minimax but useful for analysis)"""
        winner = self.check_winner(board)
        if winner == 'O':
            return 1
        elif winner == 'X':
            return -1
        else:
            return 0


class TicTacToeAI:
    def __init__(self, use_alpha_beta: bool = True):
        self.ai = MinimaxAI(use_alpha_beta)
        self.board = [' '] * 9
        self.human_player = 'X'
        self.ai_player = 'O'
        
    def print_board(self) -> None:
        """Print the current board state"""
        print("\n")
        print(f" {self.board[0]} | {self.board[1]} | {self.board[2]} ")
        print("---|---|---")
        print(f" {self.board[3]} | {self.board[4]} | {self.board[5]} ")
        print("---|---|---")
        print(f" {self.board[6]} | {self.board[7]} | {self.board[8]} ")
        print("\n")
    
    def make_human_move(self, position: int) -> bool:
        """Make human move"""
        if 0 <= position <= 8 and self.board[position] == ' ':
            self.board[position] = self.human_player
            return True
        return False
    
    def make_ai_move(self) -> None:
        """Make AI move using minimax"""
        best_move = self.ai.get_best_move(self.board)
        if best_move is not None:
            self.board[best_move] = self.ai_player
            print(f"AI plays position {best_move + 1}")
    
    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        """Check if game is over"""
        winner = self.ai.check_winner(self.board)
        if winner:
            return True, winner
        elif self.ai.is_board_full(self.board):
            return True, 'Draw'
        return False, None
    
    def reset_game(self) -> None:
        """Reset the game"""
        self.board = [' '] * 9
        self.ai.nodes_explored = 0
    
    def get_board_positions(self) -> List[int]:
        """Get valid board positions for user input"""
        return [i + 1 for i, cell in enumerate(self.board) if cell == ' ']
