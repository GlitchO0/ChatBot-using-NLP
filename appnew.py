from flask import Flask, request, jsonify
import random
import json
import torch
from nltk_utlis import tokenize, bag_pf_words
from model import NeuralNet
nltk.download('punkt')
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

bot_name = "Khadma chatbot"  # Bot's name

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_pf_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return random.choice([
        "I do not understand., Be more precise...",
        "sir! what do you want to ask about?",
        "sir! are you okay",
        "I don't understand...!",
        "what?"
    ])

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    response = get_response(message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
