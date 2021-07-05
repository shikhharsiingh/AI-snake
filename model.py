import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_hidden = 1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        return x

    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path) 

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.stat_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, nxt_state, done):
        state = torch.tensor(state, dtype = torch.float)
        nxt_state = torch.tensor(nxt_state, dtype = torch.float)
        action = torch.tensor(action, dtpe = torch.long)
        reward = torch.tensor(reward, torch.float)
        #(n, x)
        if len(state.shape) == 1:
            #(1, x)
            state = torch.unsqueeze(state, 0)
            nxt_state = torch.unsqueeze(nxt_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        #1 : predicted Q value with current state
        pred = self.model(state)

        #2: r + y * max(nxt_predicted Q value) 



