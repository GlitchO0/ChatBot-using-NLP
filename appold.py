from flask import Flask, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utlis import tokenize, bag_pf_words
import nltk
nltk.download('punkt')
import difflib

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('augmented_intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size1 = data["hidden_size1"]
hidden_size2 = data["hidden_size2"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "sam"

@app.route('/')
def home():
    return 'Chatbot API is running!'

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['message']

    sentence = tokenize(user_input)
    x = bag_pf_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.5:
        max_similarity = 0
        most_similar_response = None

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

        if most_similar_response and max_similarity > 0.25:
            bot_response = most_similar_response
        else:
            bot_response = random.choice(intent['responses'])
    else:
        bot_response = "I do not understand. Please be more precise."

    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
