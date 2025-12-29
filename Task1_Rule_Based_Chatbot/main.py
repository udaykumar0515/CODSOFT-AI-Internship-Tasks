#!/usr/bin/env python3
"""
Rule-Based Chatbot - Task 1
CODSOFT AI Internship

A simple chatbot using pattern matching and rule-based responses.
"""

from chatbot_logic import RuleBasedChatbot


def main():
    """Main function to run the chatbot"""
    print("=" * 60)
    print("🤖 RULE-BASED CHATBOT - CODSOFT AI INTERNSHIP")
    print("=" * 60)
    print("\nWelcome to the Rule-Based Chatbot!")
    print("This chatbot uses pattern matching to respond to your messages.")
    print("\nFeatures:")
    print("- Greeting responses")
    print("- Basic conversation handling")
    print("- Fallback responses for unknown inputs")
    print("- Simple pattern matching using regular expressions")
    print("\nType 'bye', 'exit', or 'quit' to end the conversation.")
    print("-" * 60)
    
    # Create and run the chatbot
    chatbot = RuleBasedChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()
