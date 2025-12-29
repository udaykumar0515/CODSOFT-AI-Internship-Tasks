import re
import random
from typing import Dict, List, Tuple


class RuleBasedChatbot:
    def __init__(self):
        self.patterns = {
            r'(?i)(hi|hello|hey|good morning|good afternoon|good evening)': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! How are you doing?"
            ],
            r'(?i)(how are you|how do you do)': [
                "I'm doing great, thank you for asking!",
                "I'm functioning perfectly! How about you?",
                "All systems operational! How can I assist you?"
            ],
            r'(?i)(what is your name|who are you)': [
                "I'm a rule-based chatbot created for the CODSOFT AI internship.",
                "I'm a simple chatbot created for the CODSOFT AI internship.",
                "I'm a simple chatbot using pattern matching to respond."
            ],
            r'(?i)(what can you do|help me|capabilities)': [
                "I can have basic conversations, answer simple questions, and provide information.",
                "I can help with greetings, basic queries, and general conversation.",
                "I'm designed for simple chat interactions and basic assistance."
            ],
            r'(?i)(bye|goodbye|see you|exit|quit)': [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! It was nice talking to you!"
            ],
            r'(?i)(thank you|thanks|thx)': [
                "You're welcome!",
                "Happy to help!",
                "No problem at all!"
            ],
            r'(?i)(weather|temperature)': [
                "I don't have access to real-time weather data, but you can check weather apps or websites.",
                "For weather information, I'd recommend checking a weather service.",
                "I can't provide weather updates, but local weather apps can help!"
            ],
            r'(?i)(time|current time)': [
                "I don't have access to real-time time data. Please check your device's clock.",
                "For the current time, please look at your system clock or phone.",
                "I can't provide real-time information, including the current time."
            ],
            r'(?i)(age|how old are you)': [
                "I'm a program, so I don't have an age in the traditional sense!",
                "Age doesn't apply to AI programs like me!",
                "I'm as old as the code that runs me!"
            ]
        }
        
        self.fallback_responses = [
            "I'm not sure how to respond to that. Can you try rephrasing?",
            "I don't understand. Could you ask something else?",
            "That's beyond my current capabilities. Try asking about something else.",
            "I'm designed for simpler conversations. Can you try a different question?",
            "I'm not equipped to handle that. What else can I help you with?"
        ]

    def get_response(self, user_input: str) -> str:
        """Generate response based on pattern matching"""
        user_input = user_input.strip()
        
        if not user_input:
            return "Please say something! I'm here to chat."
        
        for pattern, responses in self.patterns.items():
            if re.search(pattern, user_input):
                return random.choice(responses)
        
        return random.choice(self.fallback_responses)

    def chat(self) -> None:
        """Main chat loop"""
        print("🤖 Rule-Based Chatbot")
        print("Type 'bye', 'exit', or 'quit' to end the conversation.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("You: ")
                if re.search(r'(?i)(bye|goodbye|see you|exit|quit)', user_input):
                    print(f"Bot: {random.choice(self.patterns[r'(?i)(bye|goodbye|see you|exit|quit)'])}")
                    break
                
                response = self.get_response(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"Bot: Something went wrong. Let's try again!")


def main():
    bot = RuleBasedChatbot()
    bot.chat()


if __name__ == "__main__":
    main()
