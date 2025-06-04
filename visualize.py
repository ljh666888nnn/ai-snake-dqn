import matplotlib.pyplot as plt
from IPython import display
import os
import numpy as np
import pygame

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 关闭交互模式，只保存不显示
plt.ioff()  # 关闭交互模式

class VisualizeTraining:
    def __init__(self, save_dir='./plots', show_plots=False):
        self.save_dir = save_dir
        self.show_plots = show_plots  # 是否显示图形
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 用于存储数据
        self.scores = []
        self.mean_scores = []
        self.losses = []
        self.epsilons = []
        self.snake_lengths = []  # 添加蛇长度数据
        self.mean_lengths = []   # 添加平均长度数据
        self.total_steps = 0
        
    def update(self, score, mean_score, snake_length=None, mean_length=None, loss=None, epsilon=None):
        # 更新数据
        self.scores.append(score)
        self.mean_scores.append(mean_score)
        
        if snake_length is not None:
            self.snake_lengths.append(snake_length)
        
        if mean_length is not None:
            self.mean_lengths.append(mean_length)
            
        if loss is not None:
            self.losses.append(loss)
            
        if epsilon is not None:
            self.epsilons.append(epsilon)
        
        # 只有在需要时才显示图表
        if self.show_plots:
            self._plot()
        
        # 更新步数
        self.total_steps += 1
    
    def _plot(self):
        try:
            if self.show_plots:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            plt.clf()
            
            # 创建一个包含分数和蛇长的综合图表
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.title('训练进度 - 分数')
            plt.xlabel('游戏数')
            plt.ylabel('分数')
            plt.plot(self.scores, label='分数', color='blue')
            plt.plot(self.mean_scores, label='平均分数', color='red')
            plt.legend()
            
            # 如果有蛇长数据，显示蛇长图表
            if self.snake_lengths and self.mean_lengths:
                plt.subplot(2, 1, 2)
                plt.title('训练进度 - 蛇长')
                plt.xlabel('游戏数')
                plt.ylabel('蛇长')
                plt.plot(self.snake_lengths, label='蛇长', color='green')
                plt.plot(self.mean_lengths, label='平均蛇长', color='purple')
                plt.ylim(ymin=3)  # 蛇的初始长度为3
                plt.legend()
                
            if self.show_plots:
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(.1)
        except Exception as e:
            print(f"绘图时出错: {e}")
    
    def save_plot(self, game_number=None, filename='training_plot.png', include_length=False):
        try:
            # 始终使用相同的文件名，覆盖原有文件
            filename = 'training_plot.png'
                
            # 创建一个包含分数和蛇长的综合图表
            plt.figure(figsize=(12, 8))
            
            # 分数图表
            plt.subplot(2, 1, 1)
            plt.title('训练进度 - 分数')
            plt.xlabel('游戏数')
            plt.ylabel('分数')
            plt.plot(self.scores, label='分数', color='blue')
            plt.plot(self.mean_scores, label='平均分数', color='red')
            plt.legend()
            
            # 蛇长图表(如果有数据)
            if include_length and self.snake_lengths:
                plt.subplot(2, 1, 2)
                plt.title('训练进度 - 蛇长')
                plt.xlabel('游戏数')
                plt.ylabel('蛇长')
                plt.plot(self.snake_lengths, label='蛇长', color='green')
                plt.plot(self.mean_lengths, label='平均蛇长', color='purple')
                plt.ylim(ymin=3)  # 蛇的初始长度为3
                plt.legend()
            
            plt.tight_layout()
            
            # 保存综合图表
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"训练图表已保存到 {save_path}")
            
            # 如果有损失数据，也保存损失图表(覆盖原有文件)
            if self.losses:
                plt.figure(figsize=(12, 6))
                plt.title('训练损失')
                plt.xlabel('训练步骤')
                plt.ylabel('损失')
                plt.plot(self.losses)
                loss_path = os.path.join(self.save_dir, 'loss_plot.png')
                plt.savefig(loss_path)
                plt.close()
                print(f"损失图表已保存到 {loss_path}")
                
            # 如果有epsilon数据，也保存epsilon图表(覆盖原有文件)
            if self.epsilons:
                plt.figure(figsize=(12, 6))
                plt.title('探索率(Epsilon)变化')
                plt.xlabel('游戏数')
                plt.ylabel('Epsilon')
                plt.plot(self.epsilons)
                epsilon_path = os.path.join(self.save_dir, 'epsilon_plot.png')
                plt.savefig(epsilon_path)
                plt.close()
                print(f"Epsilon图表已保存到 {epsilon_path}")
        except Exception as e:
            print(f"保存图表时出错: {e}")
            
    def close(self):
        plt.close('all')


# 用于渲染游戏过程的可视化工具
class GameRender:
    def __init__(self, max_frames=1000, save_dir='./game_frames'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.frames = []
        self.max_frames = max_frames
        
    def capture_frame(self, game):
        """捕获游戏画面并保存"""
        if len(self.frames) < self.max_frames:
            # 从Pygame显示中获取当前屏幕
            frame = pygame.surfarray.array3d(game.display)
            # 保存帧
            self.frames.append(frame)
            
    def save_video(self, filename='game_video.mp4', fps=20):
        """将捕获的帧保存为视频"""
        try:
            import cv2
            
            if not self.frames:
                print("没有捕获到帧!")
                return
                
            height, width = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(os.path.join(self.save_dir, filename), fourcc, float(fps), (width, height))
            
            for frame in self.frames:
                # OpenCV使用BGR而Pygame使用RGB，需要转换
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame)
                
            video.release()
            print(f"视频已保存到 {os.path.join(self.save_dir, filename)}")
            
        except ImportError:
            print("保存视频需要安装OpenCV库 (pip install opencv-python)")
            
            # 退而求其次，保存一些关键帧
            self._save_key_frames()
        except Exception as e:
            print(f"保存视频时出错: {e}")
            # 尝试保存关键帧
            self._save_key_frames()
    
    def _save_key_frames(self, max_frames=20):
        """保存关键帧为图像"""
        if not self.frames:
            return
            
        try:
            import imageio
            
            # 选择均匀分布的关键帧
            num_frames = min(max_frames, len(self.frames))
            indices = np.linspace(0, len(self.frames)-1, num_frames, dtype=int)
            
            for i, idx in enumerate(indices):
                frame = self.frames[idx]
                imageio.imwrite(os.path.join(self.save_dir, f'frame_{i:03d}.png'), frame)
                
            print(f"已保存 {num_frames} 个关键帧到 {self.save_dir}")
        except ImportError:
            print("保存图像需要安装imageio库 (pip install imageio)")
            
            # 最基本的保存方法 - 使用matplotlib
            try:
                import matplotlib.pyplot as plt
                
                num_frames = min(max_frames, len(self.frames))
                indices = np.linspace(0, len(self.frames)-1, num_frames, dtype=int)
                
                for i, idx in enumerate(indices):
                    frame = self.frames[idx]
                    plt.imsave(os.path.join(self.save_dir, f'frame_{i:03d}.png'), frame)
                    
                print(f"已使用matplotlib保存 {num_frames} 个关键帧到 {self.save_dir}")
            except Exception as e:
                print(f"保存关键帧失败: {e}")
    
    def clear(self):
        """清除所有捕获的帧"""
        self.frames = []
        print("已清除所有帧") 