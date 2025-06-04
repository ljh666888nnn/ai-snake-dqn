import torch
import random
import numpy as np
from collections import deque
from model import LinearQNet, DQNNet
import torch.optim as optim
import torch.nn.functional as F

MAX_MEMORY = 150_000  # 增加经验池大小
BATCH_SIZE = 128  # 增加批次大小，提高训练效率
LR = 0.0003  # 进一步降低学习率，提高稳定性

class Agent:
    def __init__(self, model_type="linear", input_size=16, hidden_size=512, output_size=3):  # 增加隐藏层大小
        self.n_games = 0
        self.epsilon = 0  # 贪婪策略参数
        self.gamma = 0.98  # 提高gamma，更重视长期奖励
        self.memory = deque(maxlen=MAX_MEMORY)  # 经验回放队列
        
        # 创建模型
        if model_type == "linear":
            self.model = LinearQNet(input_size, hidden_size, output_size)
        else:
            self.model = DQNNet(input_size, hidden_size, output_size)
        
        # 创建目标网络（用于Double DQN）
        if model_type == "linear":
            self.target_model = LinearQNet(input_size, hidden_size, output_size)
        else:
            self.target_model = DQNNet(input_size, hidden_size, output_size)
        
        # 初始化目标网络权重与主网络相同
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 使用Adam优化器，降低学习率
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-6)
        
        # 添加学习率调度器，随着训练进行逐渐降低学习率
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # 优先经验回放参数
        self.alpha = 0.6  # 优先级权重
        self.beta = 0.4   # 重要性采样权重
        self.beta_increment = 0.001
        
        # 如果有GPU，使用GPU训练
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        print(f"使用设备: {self.device}")
        print(f"初始学习率: {LR}")
        print(f"模型隐藏层大小: {hidden_size}")

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        # 计算TD误差作为优先级
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float).to(self.device).unsqueeze(0)
            
            current_q = self.model(state_tensor)
            next_q = self.target_model(next_state_tensor)
            
            action_idx = np.argmax(action)
            target_q = reward if done else reward + self.gamma * torch.max(next_q).item()
            td_error = abs(target_q - current_q[0][action_idx].item())
        
        # 将经验保存到经验回放队列，带有优先级
        priority = (td_error + 1e-6) ** self.alpha
        self.memory.append((state, action, reward, next_state, done, priority))

    def train_long_memory(self):
        # 优先经验回放采样
        if len(self.memory) > BATCH_SIZE:
            # 按优先级采样
            priorities = np.array([item[5] for item in self.memory])
            probabilities = priorities / np.sum(priorities)
            
            indices = np.random.choice(len(self.memory), BATCH_SIZE, p=probabilities)
            mini_sample = [self.memory[i] for i in indices]
            
            # 重要性采样权重
            weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
            weights = weights / np.max(weights)  # 归一化
            
            self.beta = min(1.0, self.beta + self.beta_increment)  # 逐渐增加beta
        else:
            mini_sample = list(self.memory)
            weights = np.ones(len(mini_sample))

        states, actions, rewards, next_states, dones, _ = zip(*mini_sample)
        return self.train_step(states, actions, rewards, next_states, dones, weights)

    def train_step(self, states, actions, rewards, next_states, dones, weights=None):
        # 将数据转换为张量并移动到适当的设备
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        # 奖励裁剪，防止奖励过大导致训练不稳定
        rewards = torch.clamp(rewards, -30, 30)

        # Double DQN实现
        current_q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_q_targets = self.target_model(next_states)

        # 获取当前Q值
        current_q = current_q_values.gather(1, torch.argmax(actions, dim=1).unsqueeze(1)).squeeze(1)

        # 计算目标Q值（Double DQN）
        next_actions = torch.argmax(next_q_values, dim=1)
        next_q = next_q_targets.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + (self.gamma * next_q * ~dones)

        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        
        # 应用重要性采样权重
        if weights is not None:
            loss = loss * weights
        
        loss = loss.mean()

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 定期更新目标网络
        if self.n_games % 50 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # 每200步更新一次学习率
        if self.n_games % 200 == 0:
            self.scheduler.step()
        
        return loss.item()

    def get_action(self, state):
        # 改进的探索策略 - 针对突破高分优化
        if self.n_games < 500:
            # 前500局保持较高探索率
            self.epsilon = max(85 - self.n_games * 0.1, 30)
        elif self.n_games < 1500:
            # 500-1500局逐渐降低探索率
            self.epsilon = max(30 - (self.n_games - 500) * 0.015, 15)
        elif self.n_games < 3000:
            # 1500-3000局保持适中探索率，为突破高分保留探索
            self.epsilon = max(15 - (self.n_games - 1500) * 0.005, 8)
        elif self.n_games < 4000:
            # 3000-4000局保持一定探索率
            self.epsilon = max(8 - (self.n_games - 3000) * 0.003, 5)
        else:
            # 4000局后仍保持最低探索率，避免陷入局部最优
            self.epsilon = max(5 - (self.n_games - 4000) * 0.001, 3)
            
        # 获取当前游戏状态信息
        game = self._get_game_from_state(state)
        
        # 获取所有可能的动作及其安全性
        safe_actions = self._get_safe_actions(game)
        
        final_move = [0, 0, 0]
        
        # 如果有安全动作可选
        if safe_actions:
            # 在前1000局，增加食物导向的概率
            food_seeking_probability = 40 if self.n_games < 1000 else 20
            
            # 随机探索vs食物导向vs模型预测 - 三重策略
            rand_val = random.randint(0, 200)
            
            if rand_val < self.epsilon:
                # 随机探索（但只从安全动作中选择）
                move = random.choice(safe_actions)
                final_move[move] = 1
            elif rand_val < self.epsilon + food_seeking_probability:
                # 食物导向策略：选择最接近食物的安全动作
                best_food_action = self._get_food_seeking_action(game, safe_actions)
                final_move[best_food_action] = 1
            else:
                # 使用模型预测，但只考虑安全动作
                state0 = torch.tensor(state, dtype=torch.float).to(self.device)
                with torch.no_grad():
                    prediction = self.model(state0.unsqueeze(0))
                
                # 获取安全动作中Q值最高的
                best_safe_action = self._get_best_safe_action(prediction[0], safe_actions)
                final_move[best_safe_action] = 1
        else:
            # 如果没有安全动作（被困），选择延迟时间最长的动作
            print(f"⚠️  警告：第{self.n_games}局被困，选择最佳延迟动作")
            best_delay_action = self._get_best_delay_action(game)
            final_move[best_delay_action] = 1
            
        return final_move
    
    def _get_game_from_state(self, state):
        """从状态中重建游戏信息"""
        # 这里需要一个简单的方法来获取游戏对象
        # 我们需要在Agent中存储游戏引用
        if not hasattr(self, '_current_game'):
            return None
        return self._current_game
    
    def set_current_game(self, game):
        """设置当前游戏引用，用于安全检查"""
        self._current_game = game
    
    def _get_safe_actions(self, game):
        """获取所有安全的动作（优化版：更积极的安全检查）"""
        if game is None:
            return [0, 1, 2]  # 如果无法获取游戏状态，返回所有动作
        
        safe_actions = []
        
        # 测试每个动作的安全性
        for action in [0, 1, 2]:  # [直走, 右转, 左转]
            if self._is_action_safe_optimized(game, action):
                safe_actions.append(action)
        
        # 如果没有绝对安全的动作，降低安全标准
        if not safe_actions:
            for action in [0, 1, 2]:
                if self._is_action_minimally_safe(game, action):
                    safe_actions.append(action)
        
        return safe_actions if safe_actions else [0, 1, 2]  # 确保总是返回至少一个动作
    
    def _is_action_safe_optimized(self, game, action):
        """优化的安全检查：不那么保守"""
        # 模拟执行动作后的位置
        future_head = self._simulate_move(game, action)
        
        # 检查是否会撞墙
        if (future_head.x < 0 or future_head.x >= game.w or 
            future_head.y < 0 or future_head.y >= game.h):
            return False
        
        # 检查是否会撞到自己（排除尾部，因为尾部会移动）
        if future_head in game.snake[:-1]:
            return False
        
        # 只有当蛇长度较长时才进行陷阱检测
        if len(game.snake) > 8:
            if self._will_trap_snake_relaxed(game, future_head, action):
                return False
        
        return True
    
    def _is_action_minimally_safe(self, game, action):
        """最小安全检查：只避免立即死亡"""
        future_head = self._simulate_move(game, action)
        
        # 只检查是否会立即撞墙或撞到自己
        if (future_head.x < 0 or future_head.x >= game.w or 
            future_head.y < 0 or future_head.y >= game.h):
            return False
        
        if future_head in game.snake[:-1]:
            return False
        
        return True
    
    def _will_trap_snake_relaxed(self, game, future_head, action):
        """放松的陷阱检测：只在蛇很长时才严格检查"""
        from snake_game import Direction, Point, BLOCK_SIZE
        
        # 如果蛇很短，不检查陷阱
        if len(game.snake) < 10:
            return False
        
        # 简单的前瞻算法：检查移动后是否还有足够的逃脱路径
        escape_paths = self._count_escape_paths_relaxed(game, future_head)
        
        # 动态调整陷阱检测阈值 - 更宽松
        min_paths = max(2, len(game.snake) // 8)  # 降低要求
        
        if escape_paths < min_paths:
            return True
        
        return False
    
    def _count_escape_paths_relaxed(self, game, start_pos):
        """计算逃脱路径（放松版本）"""
        from snake_game import Direction, Point, BLOCK_SIZE
        
        visited = set()
        queue = [start_pos]
        reachable_count = 0
        
        # 减少搜索深度，提高效率
        max_search_depth = min(10, len(game.snake))
        
        while queue and len(visited) < max_search_depth:
            current = queue.pop(0)
            
            if current in visited:
                continue
                
            visited.add(current)
            reachable_count += 1
            
            # 检查四个方向
            for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
                next_pos = Point(current.x + dx, current.y + dy)
                
                # 检查边界
                if (next_pos.x < 0 or next_pos.x >= game.w or 
                    next_pos.y < 0 or next_pos.y >= game.h):
                    continue
                
                # 更宽松的蛇身检查
                if next_pos in game.snake[:-6]:  # 给尾部更多空间
                    continue
                    
                if next_pos not in visited:
                    queue.append(next_pos)
        
        return reachable_count
    
    def _get_food_seeking_action(self, game, safe_actions):
        """选择最接近食物的安全动作"""
        if game is None or not safe_actions:
            return safe_actions[0] if safe_actions else 0
        
        best_action = safe_actions[0]
        min_distance = float('inf')
        
        for action in safe_actions:
            # 模拟这个动作后到食物的距离
            future_head = self._simulate_move(game, action)
            distance = abs(future_head.x - game.food.x) + abs(future_head.y - game.food.y)
            
            if distance < min_distance:
                min_distance = distance
                best_action = action
        
        return best_action
    
    def _get_best_safe_action(self, q_values, safe_actions):
        """从安全动作中选择Q值最高的"""
        best_action = safe_actions[0]
        best_q = q_values[best_action].item()
        
        for action in safe_actions:
            if q_values[action].item() > best_q:
                best_q = q_values[action].item()
                best_action = action
        
        return best_action
    
    def _get_best_delay_action(self, game):
        """当所有动作都危险时，选择能延迟死亡最久的动作"""
        max_survival_steps = 0
        best_action = 0
        
        for action in [0, 1, 2]:
            survival_steps = self._calculate_survival_steps(game, action)
            if survival_steps > max_survival_steps:
                max_survival_steps = survival_steps
                best_action = action
        
        return best_action
    
    def _calculate_survival_steps(self, game, action):
        """计算某个动作能让蛇存活多少步"""
        # 简单实现：检查能在当前路径上走多远
        steps = 0
        current_head = game.head
        
        # 模拟连续执行相同动作
        for _ in range(30):  # 增加检查步数
            future_head = self._simulate_move(game, action)
            
            # 检查是否安全
            if (future_head.x < 0 or future_head.x >= game.w or 
                future_head.y < 0 or future_head.y >= game.h):
                break
            if future_head in game.snake[:-1]:
                break
                
            steps += 1
            current_head = future_head
        
        return steps
    
    def _simulate_move(self, game, action):
        """模拟移动后的蛇头位置"""
        from snake_game import Direction, Point, BLOCK_SIZE
        
        # 获取当前方向和蛇头位置
        current_direction = game.direction
        head = game.head
        
        # 根据动作计算新方向
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(current_direction)
        
        if action == 1:  # 右转
            new_direction = clock_wise[(idx + 1) % 4]
        elif action == 2:  # 左转  
            new_direction = clock_wise[(idx - 1) % 4]
        else:  # 直走
            new_direction = current_direction
        
        # 计算新的蛇头位置
        if new_direction == Direction.RIGHT:
            new_head = Point(head.x + BLOCK_SIZE, head.y)
        elif new_direction == Direction.LEFT:
            new_head = Point(head.x - BLOCK_SIZE, head.y)
        elif new_direction == Direction.DOWN:
            new_head = Point(head.x, head.y + BLOCK_SIZE)
        elif new_direction == Direction.UP:
            new_head = Point(head.x, head.y - BLOCK_SIZE)
        
        return new_head 