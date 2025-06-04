import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # 使用Xavier初始化提高训练效率
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            return True
        return False


class DQNNet(nn.Module):
    """
    更适合处理特征向量的DQN网络，针对3060 GPU优化
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # 双流架构：一个流处理状态特征，一个流处理价值估计
        self.feature_stream = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # 注意力机制，突出重要特征
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 使用Xavier初始化提高训练效率
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 注意力机制，突出重要特征
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 提取特征
        features = self.feature_stream(x)
        
        # 计算状态价值和动作优势
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 使用Dueling DQN的公式
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def save(self, file_name='dqn_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    def load(self, file_name='dqn_model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            return True
        return False 