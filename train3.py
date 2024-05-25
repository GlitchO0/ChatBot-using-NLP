import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk_utlis import bag_pf_words, tokenize, stem
from model import NeuralNet

# Load intents data from JSON file
with open('augmented_intents.json', 'r') as f:
    intents = json.load(f)

# Extract words and tags from intents
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and remove duplicates from words list
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_pf_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 500
batch_size = 8
learning_rate = 0.01
input_size = len(X_train[0])
hidden_size1 = 128
hidden_size2 = 64
output_size = len(tags)
print(input_size, output_size)

# Define custom dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create DataLoader instances for training and validation
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Convert validation data to PyTorch tensors
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# Validation dataset
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize variables for early stopping
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 3


train_accuracies = []
val_accuracies = []
from collections import Counter

tag_counts = Counter([y for x, y in xy])
print(tag_counts)

# Train the model
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Validation during training
    if (epoch + 1) % 10 == 0:
        model.eval()  # Set model to evaluation mode
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for train_inputs, train_labels in train_loader:
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)
                train_outputs = model(train_inputs)
                _, train_predicted = torch.max(train_outputs.data, 1)
                train_total += train_labels.size(0)
                train_correct += (train_predicted == train_labels).sum().item()
            train_accuracy = 100 * train_correct / train_total

        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
                val_loss += criterion(val_outputs, val_labels).item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Validation Accuracy: {val_accuracy:.2f}%, '
              f'Validation Loss: {val_loss:.4f}')

        # Append accuracies for plotting
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

# Plotting train and validation accuracies after training
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size1": hidden_size1,
    "hidden_size2": hidden_size2,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')


# Assuming you have a separate test dataset stored in X_test and y_test


