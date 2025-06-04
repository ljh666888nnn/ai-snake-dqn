import pygame
import argparse
import torch
import time
import numpy as np
from snake_game import SnakeGameAI
from agent import Agent
from visualize import GameRender
from utils import optimize_for_gpu

def play(model_path='model/dqn_model.pth', model_type='dqn', hidden_size=256, save_video=True, fps=10):
    # 针对3060 GPU进行优化
    optimize_for_gpu()
    
    # 初始化游戏
    game = SnakeGameAI()
    
    # 初始化智能体
    agent = Agent(model_type=model_type, hidden_size=hidden_size)
    
    # 加载预训练模型
    if model_type == 'dqn':
        agent.model.load(model_path.split('/')[-1])
    else:
        agent.model.load(model_path.split('/')[-1])
    
    # 初始化游戏渲染器（如果需要保存视频）
    renderer = GameRender() if save_video else None
    
    # 游戏记录
    scores = []
    
    # 主循环
    while True:
        # 获取当前状态
        state = agent.get_state(game)
        
        # 获取动作（不包含随机探索，完全基于模型）
        state_tensor = torch.tensor(state, dtype=torch.float).to(agent.device)
        prediction = agent.model(state_tensor.unsqueeze(0))
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        
        # 执行动作
        reward, done, score = game.play_step(final_move)
        
        # 捕获帧（如果需要）
        if renderer:
            renderer.capture_frame(game)
            
        # 控制FPS
        pygame.time.delay(1000 // fps)
        
        if done:
            # 游戏结束，重置游戏
            scores.append(score)
            print(f'分数: {score}')
            
            # 保存视频（如果需要）
            if renderer and save_video:
                renderer.save_video()
                renderer.clear()
            
            # 询问是否继续游戏
            font = pygame.font.SysFont('arial', 35)
            text = font.render(f'游戏结束! 分数: {score}', True, (255, 255, 255))
            text_rect = text.get_rect(center=(game.w/2, game.h/2 - 50))
            
            continue_text = font.render('按任意键继续, ESC退出', True, (255, 255, 255))
            continue_rect = continue_text.get_rect(center=(game.w/2, game.h/2 + 50))
            
            game.display.blit(text, text_rect)
            game.display.blit(continue_text, continue_rect)
            pygame.display.flip()
            
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        waiting = False
            
            game.reset()
            if renderer:
                renderer.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='让AI玩贪吃蛇')
    parser.add_argument('--model', type=str, default='model/dqn_model.pth', help='模型路径')
    parser.add_argument('--type', type=str, default='dqn', choices=['linear', 'dqn'], help='模型类型')
    parser.add_argument('--hidden', type=int, default=256, help='隐藏层大小')
    parser.add_argument('--video', action='store_true', help='保存游戏视频')
    parser.add_argument('--fps', type=int, default=10, help='游戏帧率')
    
    args = parser.parse_args()
    
    play(model_path=args.model, 
         model_type=args.type,
         hidden_size=args.hidden,
         save_video=args.video,
         fps=args.fps) 