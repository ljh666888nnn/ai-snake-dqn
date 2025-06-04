import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def plot_learning_curve(x, scores, figure_file):
    """
    绘制学习曲线
    """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('最近100场游戏的平均分数')
    plt.savefig(figure_file)

def save_scores_to_csv(scores, mean_scores, snake_lengths=None, mean_lengths=None, file_path='scores.csv'):
    """
    将分数和蛇长保存到CSV文件
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if snake_lengths is not None and mean_lengths is not None:
            writer.writerow(['Game', 'Score', 'Mean Score', 'Snake Length', 'Mean Length'])
            
            for i, (score, mean_score, length, mean_length) in enumerate(zip(scores, mean_scores, snake_lengths, mean_lengths)):
                writer.writerow([i+1, score, mean_score, length, mean_length])
        else:
            writer.writerow(['Game', 'Score', 'Mean Score'])
            
            for i, (score, mean_score) in enumerate(zip(scores, mean_scores)):
                writer.writerow([i+1, score, mean_score])
            
    print(f"数据已保存到 {file_path}")

def get_device_info():
    """
    获取当前设备信息，适用于3060 GPU调优
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = {
        "device": device,
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**2,  # MB
            "memory_reserved": torch.cuda.memory_reserved(0) / 1024**2,    # MB
            "max_memory_allocated": torch.cuda.max_memory_allocated(0) / 1024**2,  # MB
            "compute_capability": torch.cuda.get_device_capability(0)
        })
        
    return device_info

def optimize_for_gpu():
    """
    针对NVIDIA 3060进行PyTorch优化
    """
    if torch.cuda.is_available():
        # 启用CUDA异步执行
        torch.backends.cudnn.benchmark = True
        
        # 对于3060，使用确定性算法通常更快
        torch.backends.cudnn.deterministic = False
        
        # 启用cuDNN自动调优
        torch.backends.cudnn.enabled = True
        
        # 为CUDA操作设置最大内存分配
        # 3060有12GB VRAM，设置为10GB留出系统空间
        # 注意: 这仅对某些情况有效，并不保证内存使用不会超过这个值
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的可用GPU内存
        
        return True
    return False

def save_checkpoint(agent, optimizer, scores, mean_scores, snake_lengths=None, mean_lengths=None, losses=None, filename='checkpoint.pth'):
    """
    保存训练检查点
    """
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'n_games': agent.n_games,
        'scores': scores,
        'mean_scores': mean_scores,
        'losses': losses if losses else None
    }
    
    # 添加蛇长度数据
    if snake_lengths is not None and mean_lengths is not None:
        checkpoint.update({
            'snake_lengths': snake_lengths,
            'mean_lengths': mean_lengths
        })
    
    torch.save(checkpoint, filename)
    print(f"检查点已保存到 {filename}")

def load_checkpoint(agent, optimizer, filename='checkpoint.pth'):
    """
    加载训练检查点
    """
    if not os.path.isfile(filename):
        print(f"找不到检查点文件 {filename}")
        return None
        
    checkpoint = torch.load(filename)
    
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.n_games = checkpoint['n_games']
    
    print(f"已加载检查点，游戏数: {agent.n_games}")
    
    # 返回额外数据，包括蛇长度
    snake_lengths = checkpoint.get('snake_lengths', None)
    mean_lengths = checkpoint.get('mean_lengths', None)
    
    if snake_lengths is not None and mean_lengths is not None:
        return checkpoint['scores'], checkpoint['mean_scores'], checkpoint['losses'], snake_lengths, mean_lengths
    
    return checkpoint['scores'], checkpoint['mean_scores'], checkpoint['losses'] 