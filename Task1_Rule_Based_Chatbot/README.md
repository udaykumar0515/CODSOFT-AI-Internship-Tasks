# Task 1: Rule-Based Chatbot

## Overview

A simple conversational chatbot using pattern matching and rule-based responses.

## Features

- **Pattern Matching**: Uses regular expressions to understand user input
- **Greeting Responses**: Handles greetings and common phrases
- **Contextual Replies**: Provides appropriate responses based on patterns
- **Fallback Handling**: Gracefully handles unknown inputs
- **Simple & Fast**: No ML/AI training required - instant responses

## Files

- `main.py` - Entry point for the chatbot
- `chatbot_logic.py` - Core chatbot logic with pattern matching
- `examples.txt` - Example conversations
- `requirements.txt` - Dependencies (Python standard library only)

## Setup & Usage

```bash
# No installation needed - uses Python standard library only
# Python 3.7+ required

# Run the chatbot
python main.py
```

## How It Works

1. **User Input**: User types a message
2. **Pattern Matching**: Chatbot matches input against predefined patterns using regex
3. **Response Selection**: Selects appropriate response from rule-based logic
4. **Output**: Displays response to user

## Example Conversation

```
User: Hello!
Bot: Hi there! How can I help you today?

User: What's your name?
Bot: I'm a rule-based chatbot created for the CODSOFT AI Internship.

User: How are you?
Bot: I'm doing great, thanks for asking!

User: bye
Bot: Goodbye! Have a great day!
```

## Technical Details

- **Pattern Matching**: Regular expressions (re module)
- **Response Logic**: Rule-based if-else conditions
- **No Dependencies**: Pure Python standard library
- **Simple Architecture**: Easy to extend with new patterns

## Requirements Met

✓ **Rule-Based System**: Uses if-else and pattern matching  
✓ **Conversational**: Handles greetings and basic conversation  
✓ **User-Friendly**: Simple text-based interface  
✓ **Extensible**: Easy to add new patterns and responses

## Extending the Chatbot

Add new patterns in `chatbot_logic.py`:

```python
# Add new pattern
patterns = {
    r"weather": ["The weather is nice today!", "I don't have weather data."],
    r"time": ["I don't have access to the current time."]
}
```

## Notes

- Uses Python's `re` module for pattern matching
- No external dependencies required
- Responses are predefined, not generated
- Type 'bye', 'exit', or 'quit' to end conversation
