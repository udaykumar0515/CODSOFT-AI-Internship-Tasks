# CODSOFT-AI-Internship-Tasks

## Project Overview

This repository contains three AI internship tasks completed as part of the CODSOFT AI Internship program. Each task demonstrates different AI concepts and implementation approaches:

- **Task 1**: Rule-based chatbot using pattern matching
- **Task 2**: Tic-Tac-Toe AI with unbeatable Minimax algorithm
- **Task 3**: Image Captioning using pretrained CNN + RNN/Transformer

## Task Descriptions

### Task 1: Rule-Based Chatbot
A simple conversational AI that uses IF-ELSE conditions and pattern matching to respond to user inputs. Handles greetings, basic intents, and provides fallback responses.

### Task 2: Tic-Tac-Toe AI
A console-based Tic-Tac-Toe game where human players compete against an unbeatable AI opponent using the Minimax algorithm (with optional Alpha-Beta pruning).

### Task 3: Image Captioning
An image captioning system that uses pretrained CNN models (VGG/ResNet) for feature extraction and RNN/LSTM or Transformer decoders to generate descriptive captions for images.

## How to Run Each Task

### Task 1: Rule-Based Chatbot
```bash
cd Task1_Rule_Based_Chatbot
pip install -r requirements.txt
python main.py
```

### Task 2: Tic-Tac-Toe AI
```bash
cd Task2_TicTacToe_AI
pip install -r requirements.txt
python main.py
```

### Task 3: Image Captioning
```bash
cd Task3_Image_Captioning
pip install -r requirements.txt
# For inference
python inference.py
# For training (optional)
jupyter notebook training.ipynb
```

## Dependencies

Each task has its own `requirements.txt` file with specific dependencies:

- **Task 1**: Basic Python libraries (re, random)
- **Task 2**: No external dependencies required
- **Task 3**: torch, torchvision, PIL, numpy, matplotlib

## Screenshots

*Screenshots will be added here as they become available*

## Demo Instructions

1. **Chatbot Demo**: Start the chatbot and try different conversation patterns
2. **Tic-Tac-Toe Demo**: Play against the AI to experience unbeatable gameplay
3. **Image Captioning Demo**: Upload images to see generated captions

## Repository Structure

```
CODSOFT-AI-Internship-Tasks/
├── README.md
├── Task1_Rule_Based_Chatbot/
│   ├── main.py
│   ├── chatbot_logic.py
│   ├── examples.txt
│   ├── requirements.txt
│   └── README.md
├── Task2_TicTacToe_AI/
│   ├── main.py
│   ├── minimax.py
│   ├── game_logic.py
│   ├── requirements.txt
│   └── README.md
├── Task3_Image_Captioning/
│   ├── main.py
│   ├── inference.py
│   ├── model.py
│   ├── training.ipynb
│   ├── requirements.txt
│   ├── README.md
│   └── sample_images/
└── assets/
    └── screenshots/
```

## Author

CODSOFT AI Internship Program

## License

This project is for educational purposes as part of the CODSOFT AI Internship.
