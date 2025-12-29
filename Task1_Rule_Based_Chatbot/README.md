# Task 1: Rule-Based Chatbot

## Overview

A simple rule-based chatbot that uses pattern matching and regular expressions to respond to user inputs. This implementation demonstrates basic conversational AI without requiring machine learning models.

## Features

- **Pattern Matching**: Uses regular expressions for input recognition
- **Multiple Response Variations**: Random selection from predefined responses
- **Fallback Handling**: Graceful responses for unrecognized inputs
- **Conversation Flow**: Supports greetings, basic questions, and farewells
- **Error Handling**: Robust handling of edge cases and interruptions

## Supported Intents

1. **Greetings**: hi, hello, hey, good morning/afternoon/evening
2. **Well-being**: how are you, how do you do
3. **Identity**: what is your name, who are you
4. **Capabilities**: what can you do, help me
5. **Farewell**: bye, goodbye, see you, exit, quit
6. **Gratitude**: thank you, thanks, thx
7. **Information Requests**: weather, time, age
8. **Fallback**: Unknown inputs

## Files Structure

```
Task1_Rule_Based_Chatbot/
├── main.py              # Entry point and user interface
├── chatbot_logic.py     # Core chatbot implementation
├── examples.txt         # Conversation examples and documentation
├── requirements.txt     # Dependencies (standard library only)
└── README.md           # This file
```

## How to Run

1. Navigate to the Task1 directory:
   ```bash
   cd Task1_Rule_Based_Chatbot
   ```

2. Install dependencies (optional - uses standard library):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the chatbot:
   ```bash
   python main.py
   ```

4. Interact with the chatbot:
   - Type your messages and press Enter
   - Type 'bye', 'exit', or 'quit' to end the conversation
   - Use Ctrl+C to force quit

## Implementation Details

### Pattern Matching
The chatbot uses Python's `re` module for pattern matching with case-insensitive regular expressions.

### Response Generation
- Each pattern has multiple response options
- Random selection provides conversational variety
- Fallback responses handle unrecognized inputs

### Error Handling
- Graceful handling of empty inputs
- Keyboard interrupt handling
- General exception catching

## Example Conversation

```
🤖 Rule-Based Chatbot
Type 'bye', 'exit', or 'quit' to end the conversation.
--------------------------------------------------
You: hello
Bot: Hi there! What can I do for you?
You: what can you do?
Bot: I can have basic conversations, answer simple questions, and provide information.
You: thanks
Bot: You're welcome!
You: bye
Bot: Goodbye! Have a great day!
```

## Technical Specifications

- **Language**: Python 3.7+
- **Dependencies**: Standard library only (re, random, typing)
- **Pattern Matching**: Regular expressions
- **Response Strategy**: Rule-based with randomization
- **Input Handling**: Console-based text input

## Extensions (Optional)

To enhance this chatbot, consider:
- Adding more pattern-response pairs
- Implementing context awareness
- Adding response memory
- Integrating external APIs
- Supporting multiple languages
- Adding GUI interface

## Evaluation Criteria

This task demonstrates:
- ✅ Rule-based logic implementation
- ✅ Pattern matching capabilities
- ✅ Conversation flow management
- ✅ Error handling
- ✅ Clean code structure
- ✅ Documentation and examples
