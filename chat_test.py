import random
import json
import torch
from model import NeuralNet
from nltk_utlis import tokenize, bag_pf_words
from time import sleep
import difflib
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise CPU

with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load the trained model and other necessary data
FILE = "data.pt"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size1 = data["hidden_size1"]
hidden_size2 = data["hidden_size2"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size1,hidden_size2, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "sam"
print("Let's chat! Type 'quit' to exit.")

# Print all available tags
print("Categories:")
specific_intents = ["Services", "Working_Hours", "problems"]
for intent in intents["intents"]:
    if intent["tag"] in specific_intents:
        print("-", intent["tag"])

# Prompt user to choose a tag
while True:
    user_tag = input("Choose a category to start talking about: or type 'pass' to start taking: ")
    if user_tag.lower() == "quit":
        print("Sad to see you go ,wish you good repairs. ")
        sys.exit()
    elif user_tag.lower().startswith("pass"):
        print("Skipping category selection...")
        break
    # Check if the user entered a valid tag
    if user_tag.lower() in [intent["tag"].lower() for intent in intents["intents"]]:
        break
    else:
        print("Invalid category. Please choose from the listed categories.")

# Retrieve responses for the chosen tag
chosen_responses = []
for intent in intents["intents"]:
    if intent["tag"].lower() == user_tag.lower():
        chosen_responses = intent["responses"]

if chosen_responses:
    response_to_print = random.choice(chosen_responses)
    print(f"{bot_name}: {response_to_print}")
# else:
#     print("No responses found for the chosen tag.")

print("\nNow let's start chatting!")

chat_history = []
# compares the sentence with each word and if If a similarity above a certain threshold is found select it
while True:

    sentence = input('You: ')
    if sentence.lower() == "quit":
        print("Thank you for chatting with me!")
        break

    sentence = tokenize(sentence)
    x = bag_pf_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.5:  # Adjust confidence threshold as needed 0.5
        max_similarity = 0
        most_similar_response = None
# prforming the text similarity to get the most convenient words..
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                for response in intent['responses']:
                    response_words = tokenize(response)
                    for word in response_words:
                        for sentence_word in sentence:
                            similarity = difflib.SequenceMatcher(None, sentence_word, word).ratio()
                            if similarity > max_similarity:
                                max_similarity = similarity
                                most_similar_response = response

        if most_similar_response and max_similarity > 0.25: # 0.5,0.75
            print(f"{bot_name}: {most_similar_response}")
            chat_history.append((sentence, most_similar_response))
        else:
            response = random.choice(intent['responses'])
            print(f"{bot_name}: {response}")
            chat_history.append((sentence, response))
    else:
        print(f"{bot_name}: I do not understand. ,Be more precise...") #Do u need any thing else

# Ask the user to rate the service

print("Do you want to rate our service? (Press 'y' for yes or 'n' for no): ")
response = input().lower()

if response == 'y':
    print("Please rate our service from 1 to 5 (1 being the lowest and 5 being the highest): ")
    rating = input()

    while rating not in ['1', '2', '3', '4', '5']:
        print("Invalid input. Please rate our service from 1 to 5: ")
        rating = input()

    # Save the rating

    # Load existing ratings if available
    try:
        with open('ratings.json', 'r') as file:
            rates = json.load(file)
    except FileNotFoundError:
        rates = []

    # Add the new rating to the list
    rates.append(rating)

    # Save the updated ratings to the file
    with open('ratings.json', 'w') as file:
        json.dump(rates, file)

    print("Thank you for your feedback!")
elif response == 'n':
    print("Thank you for your time!")
else:
    print("Invalid input. Please respond with 'y' or 'n'.")


