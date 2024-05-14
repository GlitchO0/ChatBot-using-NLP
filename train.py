import json
import numpy as np
from model import NeuralNet
from nltk_utlis import tokenize,stem,bag_pf_words
import torch            # to create pytorch dataset
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

with open('intents.json','r')as f:
    intents=json.load(f)

#print(intents)

all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))
#print(tags)


x_train=[]
y_train=[]
for(pattern_sentence,tag) in xy:
    bag=bag_pf_words(pattern_sentence,all_words)
    x_train.append(bag)

    label=tags.index(tag)
    y_train.append(label) # CrossEntropy loss only want to have the class labels

x_train=np.array(x_train)
y_train=np.array(y_train)    # the training data


# we are gone prepare our dataset for better training
class ChatDataset(Dataset):   #create a class with new dataset
    def __init__(self):  #int function
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data =y_train
    #dataset[idx]
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples


#Hyper parameters
# Define model
class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, output_size)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            out = self.fc1(x)
            out = torch.relu(out)
            out = self.dropout(out)

            out = self.fc2(out)
            out = torch.relu(out)
            out = self.dropout(out)

            out = self.fc3(out)
            return out

# Hyperparameters
batch_size = 32
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 32
output_size = len(tags)
input_size = len(x_train[0])  # Corrected input size

# Rest of the code remains the same




# batch_size=32   #100
# hidden_size=64  #63
# output_size=len(tags)
# input_size=len(x_train[0])  #all word
learning_rate=0.001
num_epochs=1000   #1000
#
#print(input_size,len(all_words))  #input size match the all words
#print(output_size,tags)


dataset=ChatDataset()   # create object from the class
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)#num of workers for multi treading ,  # create a dataloader

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this command to use GPU otherwise use cpu

#Model

model=NeuralNet(input_size,hidden_size1,hidden_size2,output_size).to(device)

#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)





for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()
        # print("Input data shape:", words.shape)
        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100== 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

print(f'Final Loss: {loss.item():.4f}')



#to calculate the accuracy

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

#calculate the accuracy
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class labels

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy

# Usage example:
accuracy = calculate_accuracy(model, train_loader, device)
print(f'Accuracy on the test set: {accuracy * 100:.2f}%')


#save the Data

data={ "model_state": model.state_dict(),
       "input_size" : input_size,
       "output_size": output_size,
       "hidden_size1": hidden_size1,
       "hidden_size2": hidden_size2,
       "all_words"  : all_words,
       "tags"       : tags
}
FILE="data.pt"
torch.save(data,FILE)

print(f"training complete file saved to -> {FILE}")
