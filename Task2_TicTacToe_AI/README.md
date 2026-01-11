# Task 2: Tic-Tac-Toe AI

## Overview

An unbeatable Tic-Tac-Toe AI using the Minimax algorithm with Alpha-Beta pruning optimization.

## 🎥 Demo Video

Watch the AI in action: [View Demo](../demo_videos/task2.mp4)

## Features

- **Unbeatable AI**: Never loses - plays perfect moves
- **Minimax Algorithm**: Evaluates all possible game states
- **Alpha-Beta Pruning**: Optimizes search by pruning unnecessary branches
- **Human vs AI**: Play against the AI
- **AI vs AI**: Watch two AI players compete
- **Clean Interface**: Simple text-based game board

## Files

- `main.py` - Entry point for the game
- `game_logic.py` - Game board logic and main game loop (~200 lines)
- `minimax.py` - Minimax algorithm with Alpha-Beta pruning (~180 lines)
- `requirements.txt` - Dependencies (Python standard library only)

## Setup & Usage

```bash
# No installation needed - uses Python standard library only
# Python 3.7+ required

# Run the game
python main.py
```

## How to Play

1. Choose game mode:

   - **1**: Human (X) vs AI (O)
   - **2**: AI (X) vs Human (O)
   - **3**: AI vs AI (watch)

2. Enter moves using numbers 1-9:

```
 1 | 2 | 3
-----------
 4 | 5 | 6
-----------
 7 | 8 | 9
```

3. Try to get three in a row (horizontal, vertical, or diagonal)

## Minimax Algorithm

The AI uses the **Minimax algorithm** to:

1. Evaluate all possible future game states
2. Assume optimal play from both players
3. Choose the move that maximizes its chances of winning
4. Minimize the opponent's chances

**Alpha-Beta Pruning** optimization reduces:

- Unnecessary branch exploration
- Computation time
- Memory usage

## Example Game Board

```
   |   |
-----------
   | X |
-----------
 O |   |
```

## Algorithm Explained

```python
def minimax(board, depth, is_maximizing, alpha, beta):
    # Check terminal state
    if game_over:
        return score

    if is_maximizing:
        # AI's turn - maximize score
        max_eval = -infinity
        for each possible move:
            score = minimax(new_board, depth+1, False, alpha, beta)
            max_eval = max(max_eval, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # Alpha-Beta pruning
        return max_eval
    else:
        # Opponent's turn - minimize score
        # Similar logic but minimize
```

## Features Breakdown

| Feature            | Description                              |
| ------------------ | ---------------------------------------- |
| **Perfect Play**   | AI never loses (only wins or draws)      |
| **Fast Decisions** | Alpha-Beta pruning speeds up computation |
| **Multiple Modes** | Human vs AI or AI vs AI                  |
| **Depth-Limited**  | Optimized depth search                   |

## Technical Details

- **Language**: Python 3.7+
- **Algorithm**: Minimax with Alpha-Beta Pruning
- **Complexity**: O(b^d) where b=branching factor, d=depth
- **Optimization**: Alpha-Beta prunes ~50% of branches
- **No Dependencies**: Pure Python standard library

## Requirements Met

✓ **Minimax Algorithm**: Full implementation with pruning  
✓ **Optimal Play**: AI plays perfectly  
✓ **Game Logic**: Complete Tic-Tac-Toe implementation  
✓ **User Interface**: Clean text-based interface

## Game States

- **Win**: Three in a row
- **Draw**: Board full, no winner
- **In Progress**: Game continues

## Scoring System

```python
AI wins:      +10
Human wins:   -10
Draw:          0
```

## Notes

- The AI is **unbeatable** - it will never lose
- Best you can do is draw if you play perfectly
- Alpha-Beta pruning makes it very fast
- No external libraries needed
- Try different strategies to achieve a draw!

## Challenge

Can you force a draw against the AI? It's the best you can do! 🎮
