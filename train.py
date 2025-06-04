import argparse
import pygame
import numpy as np
from snake_game import SnakeGameAI
from agent import Agent
from visualize import VisualizeTraining
import torch
from collections import Counter

def train(model_type="dqn", max_games=5000, save_interval=100, show_game=False, speed=100, show_plots=False):
    """
    训练贪吃蛇AI
    """
    print(f"🚀 开始训练贪吃蛇AI")
    print(f"模型: {model_type}")
    print(f"游戏数量: {max_games}")
    print(f"保存间隔: {save_interval}")
    print(f"显示游戏: {'是' if show_game else '否'}")
    print(f"游戏速度: {speed}")
    print("-" * 50)
    
    # 初始化
    agent = Agent(model_type=model_type)
    game = SnakeGameAI(speed=speed)
    visualizer = VisualizeTraining(show_plots=show_plots)
    
    # 训练数据记录
    scores = []
    mean_scores = []
    losses = []
    epsilons = []
    snake_lengths = []
    mean_lengths = []
    death_reasons = []
    
    total_score = 0
    record = 0
    
    print(f"开始训练循环，目标{max_games}局...")
    
    for game_num in range(max_games):
        # 内部游戏循环
        while True:
            # 设置当前游戏引用，用于安全检查
            agent.set_current_game(game)
            
            # 获取旧状态
            state_old = agent.get_state(game)
            
            # 获取动作
            final_move = agent.get_action(state_old)
            
            # 执行动作并获取新状态
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            # 训练短期记忆
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # 游戏结束处理
                game.reset()
                agent.n_games += 1
                
                # 训练长期记忆
                current_loss = 0
                if len(agent.memory) > 1000:  # 确保有足够的经验
                    current_loss = agent.train_long_memory()
                    if current_loss is None:
                        current_loss = 0
                
                losses.append(current_loss)
                
                # 记录死亡原因
                if hasattr(game, 'death_reason') and game.death_reason:
                    death_reasons.append(game.death_reason)
                
                # 更新记录
                if score > record:
                    record = score
                    # 保存最佳模型
                    if model_type == "linear":
                        agent.model.save('best_linear_model.pth')
                    else:
                        agent.model.save('best_dqn_model.pth')
                
                # 记录统计数据
                total_score += score
                mean_score = total_score / agent.n_games
                current_length = 3 + score  # 初始长度3 + 分数
                
                scores.append(score)
                mean_scores.append(mean_score)
                epsilons.append(agent.epsilon)
                snake_lengths.append(current_length)
                
                if snake_lengths:
                    mean_length = sum(snake_lengths) / len(snake_lengths)
                    mean_lengths.append(mean_length)
                else:
                    mean_lengths.append(current_length)
                
                # 更新可视化器的数据
                visualizer.scores = scores
                visualizer.mean_scores = mean_scores
                visualizer.losses = losses
                visualizer.epsilons = epsilons
                visualizer.snake_lengths = snake_lengths
                visualizer.mean_lengths = mean_lengths
                
                # 定期输出进度（每50局）
                if agent.n_games % 50 == 0 or agent.n_games <= 10:
                    current_lr = agent.optimizer.param_groups[0]['lr']
                    print(f'游戏 {agent.n_games:4d}/{max_games}: 分数={score:2d}, 记录={record:2d}, '
                          f'平均分={mean_score:.2f}, 蛇长={current_length:2d}, '
                          f'平均长度={mean_length:.2f}, ε={agent.epsilon:.1f}, '
                          f'损失={current_loss:.4f}, lr={current_lr:.6f}')
                
                # 定期保存和统计
                if agent.n_games % save_interval == 0:
                    print(f'\n=== 第 {agent.n_games} 局检查点 ===')
                    
                    # 保存模型
                    if model_type == "linear":
                        agent.model.save(f'checkpoint_linear_{agent.n_games}.pth')
                    else:
                        agent.model.save(f'checkpoint_dqn_{agent.n_games}.pth')
                    
                    # 死亡原因统计
                    if death_reasons:
                        death_stats = Counter(death_reasons)
                        total_deaths = len(death_reasons)
                        print("死亡原因统计:")
                        for reason, count in death_stats.most_common():
                            percentage = (count / total_deaths) * 100
                            print(f"  {reason}: {count}次 ({percentage:.1f}%)")
                    
                    # 保存可视化 - 使用正确的方法名
                    visualizer.save_plot(game_number=agent.n_games, include_length=True)
                    
                    print(f'检查点已保存 - 平均分: {mean_score:.2f}, 最高分: {record}')
                    print('=' * 50)
                
                # 退出内部循环，进入下一局游戏
                break
    
    # 训练完成
    print('\n🎉 训练完成!')
    print(f'总游戏数: {max_games}')
    print(f'最高分: {record}')
    print(f'最终平均分: {mean_score:.2f}')
    if mean_lengths:
        print(f'最终平均蛇长: {mean_lengths[-1]:.2f}')
    
    # 最终统计
    if death_reasons:
        print("\n📊 最终死亡原因统计:")
        death_stats = Counter(death_reasons)
        total_deaths = len(death_reasons)
        for reason, count in death_stats.most_common():
            percentage = (count / total_deaths) * 100
            print(f"  {reason}: {count}次 ({percentage:.1f}%)")
    
    # 保存最终模型和可视化
    final_model_name = f'final_{model_type}_model_{max_games}games.pth'
    agent.model.save(final_model_name)
    print(f"\n💾 最终模型已保存: {final_model_name}")
    
    # 最终可视化 - 使用正确的方法名
    visualizer.save_plot(filename='final_training_plot.png', include_length=True)
    
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练贪吃蛇AI')
    parser.add_argument('--model', type=str, default='dqn', choices=['linear', 'dqn'], 
                        help='模型类型: linear 或 dqn (默认: dqn)')
    parser.add_argument('--games', type=int, default=5000, 
                        help='训练游戏数量 (默认: 5000)')
    parser.add_argument('--interval', type=int, default=100, 
                        help='保存间隔 (默认: 100)')
    parser.add_argument('--show', action='store_true', 
                        help='显示游戏界面')
    parser.add_argument('--no-show', action='store_true', 
                        help='不显示游戏界面 (默认)')
    parser.add_argument('--speed', type=int, default=100, 
                        help='游戏速度 (默认: 100)')
    parser.add_argument('--plots', action='store_true',
                        help='显示实时图表')
    
    args = parser.parse_args()
    
    # 确定是否显示游戏
    show_game = args.show and not args.no_show
    
    train(
        model_type=args.model,
        max_games=args.games,
        save_interval=args.interval,
        show_game=show_game,
        speed=args.speed,
        show_plots=args.plots
    ) 