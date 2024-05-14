import torch
import torch.nn as nn

# class NeuralNet(nn.Module):
#     def __init__(self,input_size,hidden_size,num_classes): #feed-forward neural net
#         super(NeuralNet,self).__init__()
#         self.l1=nn.Linear(input_size,hidden_size)
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.l3 = nn.Linear(hidden_size, num_classes)
#         self.relu=nn.ReLU()
#
#
#     def forward(self,x):
#         out=self.l1(x)
#         out=self.relu(out)
#         out = self.l2(x)
#         out = self.relu(out)
#         out = self.l3(x)
#         out = self.relu(out)
#         return out


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


# Rest of the code remains the same
