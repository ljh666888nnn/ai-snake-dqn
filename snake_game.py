import pygame
import numpy as np
import random
from enum import Enum
from collections import namedtuple
import os
import math

# 定义方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 定义坐标点
Point = namedtuple('Point', 'x, y')

# 定义颜色
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 180, 0)
GREEN2 = (0, 150, 0)
BLACK = (0, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

# 定义方块大小和游戏速度
BLOCK_SIZE = 20
SPEED = 20

class SnakeGameAI:
    def __init__(self, w=640, h=480, speed=SPEED):
        self.w = w
        self.h = h
        # 初始化游戏状态
        self.display = None
        self.clock = None
        self.speed = speed
        self.font = None
        self.prev_distance = 0  # 记录上一步到食物的距离
        self.same_direction_count = 0  # 记录连续同方向移动的次数
        self.no_progress_count = 0  # 记录没有接近食物的次数
        self.last_positions = []  # 记录最近的位置，用于检测循环
        self.death_reason = None  # 记录死亡原因
        self.reset()
        
    def reset(self):
        # 初始化蛇的位置和方向
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                     Point(self.head.x-BLOCK_SIZE, self.head.y),
                     Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.death_reason = None  # 重置死亡原因
        
        # 重置奖励相关状态
        self.prev_distance = self._get_distance_to_food()
        self.same_direction_count = 0
        self.no_progress_count = 0
        self.last_positions = []
        
        # 用于视觉显示的初始化
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('AI贪吃蛇')
            self.clock = pygame.time.Clock()
            
            # 设置中文字体
            try:
                # 尝试使用系统中的中文字体
                self.font = pygame.font.SysFont('simhei', 25)
            except:
                # 如果没有找到合适的中文字体，使用默认字体
                self.font = pygame.font.SysFont('arial', 25)
                print("警告：未找到适合显示中文的字体，可能导致中文显示不正确")
        
        return self.get_state()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def _get_distance_to_food(self):
        """计算蛇头到食物的曼哈顿距离"""
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    
    def _is_circle_movement(self):
        """检测是否在原地打转"""
        # 保存最近的10个位置
        if len(self.last_positions) > 20:
            self.last_positions.pop(0)
        
        # 如果同一个位置出现多次，可能在打转
        position_counts = {}
        for pos in self.last_positions:
            if pos in position_counts:
                position_counts[pos] += 1
            else:
                position_counts[pos] = 1
                
        # 如果某个位置出现3次以上，认为在打转
        for count in position_counts.values():
            if count >= 3:
                return True
        return False
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. 收集用户输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. 移动
        prev_head = self.head  # 记录移动前的位置
        self._move(action) # 更新蛇头
        self.snake.insert(0, self.head)
        
        # 记录位置历史
        self.last_positions.append(self.head)
        
        # 3. 计算奖励和检查游戏是否结束
        reward = 0
        game_over = False
        
        # 撞墙或撞到自己，游戏结束，给予负奖励
        collision_result = self.is_collision()
        if collision_result[0]:
            game_over = True
            self.death_reason = collision_result[1]  # 记录死亡原因
            
            # 根据分数调整死亡惩罚，高分时死亡惩罚更重
            base_penalty = -15
            score_penalty = -self.score * 0.5  # 分数越高，死亡损失越大
            
            if self.death_reason == "自杀":
                reward = base_penalty + score_penalty - 5  # 自杀额外惩罚
            else:
                reward = base_penalty + score_penalty  # 撞墙惩罚
            
            return reward, game_over, self.score
        
        # 超过最大帧数，游戏结束
        max_frames = 200 * len(self.snake)  # 增加最大帧数限制
        if self.frame_iteration > max_frames:
            game_over = True
            self.death_reason = "超时"
            reward = -10 - self.score * 0.3  # 超时惩罚也考虑分数
            return reward, game_over, self.score
        
        # 4. 放置新食物或者移动
        if self.head == self.food:
            # 吃到食物的奖励策略优化
            base_food_reward = 25  # 进一步增加基础奖励
            
            # 长度奖励：蛇越长，吃食物奖励越高
            length_bonus = len(self.snake) * 0.8
            
            # 效率奖励：步数越少吃到食物，奖励越高
            efficiency_bonus = max(0, 80 - self.frame_iteration) * 0.15
            
            reward = base_food_reward + length_bonus + efficiency_bonus
            
            self.score += 1
            self._place_food()
            
            # 重置相关计数器
            self.no_progress_count = 0
            self.same_direction_count = 0
            self.frame_iteration = 0  # 重置帧计数，给下次吃食物更多时间
        else:
            self.snake.pop()
            
            # 距离奖励优化 - 增强食物导向性
            current_distance = self._get_distance_to_food()
            distance_change = self.prev_distance - current_distance
            
            # 更强的距离奖励/惩罚
            if distance_change > 0:
                # 接近食物，给予更强的奖励
                reward += distance_change * 0.08  # 从0.02增加到0.08
                self.no_progress_count = 0
            elif distance_change < 0:
                # 远离食物，给予更强的惩罚
                reward += distance_change * 0.12  # 从0.03增加到0.12
                self.no_progress_count += 1
            else:
                # 距离不变，惩罚
                reward -= 0.3  # 从0.1增加到0.3
                self.no_progress_count += 1
            
            # 更新前一步距离
            self.prev_distance = current_distance
            
            # 长时间没有进展的强化惩罚
            if self.no_progress_count > 20:  # 降低阈值从30到20
                reward -= 3  # 增加惩罚
                if self.no_progress_count > 35:  # 降低阈值从50到35
                    reward -= 8  # 严重拖沓惩罚
                    
            # 循环移动惩罚 - 增强
            circle_penalty = self._get_circle_penalty()
            reward += circle_penalty
                
            # 存活奖励：奖励AI保持存活状态，但降低基础奖励
            survival_reward = 0.02 + len(self.snake) * 0.005  # 降低存活奖励，避免过度保守
            reward += survival_reward
            
            # 空间利用奖励
            space_reward = self._calculate_space_utilization_reward()
            reward += space_reward
            
            # 安全移动奖励：检查当前位置的安全性
            safety_bonus = self._calculate_safety_bonus()
            reward += safety_bonus
            
            # 食物方向奖励：奖励朝向食物的移动
            direction_reward = self._calculate_food_direction_reward(prev_head)
            reward += direction_reward
        
        # 5. 更新UI和时钟
        self._update_ui()
        self.clock.tick(self.speed)
        
        # 6. 返回游戏状态
        return reward, game_over, self.score
    
    def _warn_about_potential_suicide(self, reward):
        """检查下一步可能的动作是否会导致自杀，并给予警告性惩罚"""
        # 获取下一步的潜在位置
        next_positions = self._get_next_positions()
        
        # 检查这些位置是否会导致碰撞
        danger_positions = []
        for pos in next_positions:
            if self._will_collide_with_body(pos):
                danger_positions.append(pos)
        
        # 如果有危险位置，给予额外惩罚
        if len(danger_positions) > 2:  # 如果多个方向都是危险的
            reward -= 5.0  # 严重警告
        elif len(danger_positions) == 2:  # 如果两个方向是危险的
            reward -= 3.0  # 中等警告
        elif len(danger_positions) == 1:  # 如果一个方向是危险的
            reward -= 1.5  # 轻微警告
            
        return reward  # 修复Bug：返回修改后的奖励值
    
    def _calculate_safety_bonus(self):
        """计算安全移动的奖励"""
        safety_bonus = 0
        
        # 计算当前位置周围的安全空间
        safe_directions = 0
        head = self.head
        
        # 检查四个方向的安全性
        directions = [
            Point(head.x + BLOCK_SIZE, head.y),  # 右
            Point(head.x - BLOCK_SIZE, head.y),  # 左
            Point(head.x, head.y + BLOCK_SIZE),  # 下
            Point(head.x, head.y - BLOCK_SIZE)   # 上
        ]
        
        for pos in directions:
            if not self.is_collision(pos)[0]:
                safe_directions += 1
        
        # 根据安全方向数量给予奖励
        if safe_directions >= 3:
            safety_bonus = 0.5
        elif safe_directions == 2:
            safety_bonus = 0.2
        elif safe_directions == 1:
            safety_bonus = -0.3  # 只有一个安全方向，给轻微惩罚
        else:
            safety_bonus = -1.0  # 无路可走，严重惩罚
            
        return safety_bonus
    
    def _get_next_positions(self):
        """获取下一步可能的位置"""
        head = self.head
        positions = []
        
        # 向右移动
        positions.append(Point(head.x + BLOCK_SIZE, head.y))
        # 向左移动
        positions.append(Point(head.x - BLOCK_SIZE, head.y))
        # 向上移动
        positions.append(Point(head.x, head.y - BLOCK_SIZE))
        # 向下移动
        positions.append(Point(head.x, head.y + BLOCK_SIZE))
        
        return positions
    
    def _will_collide_with_body(self, point):
        """检查给定点是否会与蛇身碰撞"""
        # 排除蛇尾，因为它会在下一步移动
        return point in self.snake[:-1]
    
    def is_collision(self, pt=None):
        """
        检查是否发生碰撞
        返回: (bool, 原因) - 是否碰撞及碰撞原因
        """
        if pt is None:
            pt = self.head
            
        # 检查是否撞到自己（自杀）
        if pt in self.snake[1:]:
            return True, "自杀"
            
        # 检查是否撞墙
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True, "撞墙"
        
        return False, None
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # 绘制蛇
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        # 绘制食物
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 绘制分数和蛇长
        score_text = self.font.render(f"分数: {self.score}", True, WHITE)
        self.display.blit(score_text, [10, 10])
        
        # 添加蛇的长度显示 - 正确显示为初始长度3加上得分
        length_text = self.font.render(f"蛇长: {3 + self.score}", True, WHITE)
        self.display.blit(length_text, [10, 40])
        
        # 如果有死亡原因，显示在屏幕上
        if self.death_reason:
            death_text = self.font.render(f"死亡原因: {self.death_reason}", True, RED)
            self.display.blit(death_text, [10, 70])
        
        pygame.display.flip()
        
    def _move(self, action):
        # [直走, 右转, 左转]
        
        # 获取当前方向的时钟方向顺序
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        prev_direction = self.direction
        
        if np.array_equal(action, [1, 0, 0]):
            # 保持当前方向
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # 右转 (顺时针)
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1]
            # 左转 (逆时针)
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        # 检查是否连续同方向移动
        if new_dir == prev_direction:
            self.same_direction_count += 1
        else:
            self.same_direction_count = 0
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
    
    def get_state(self):
        """
        获取游戏状态作为AI的输入
        扩展状态表示，增加更多有用信息
        """
        head = self.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # 扩展状态表示 - 现在有16个特征
        state = [
            # 前方、右侧、左侧危险检测
            (dir_r and self.is_collision(point_r)[0]) or 
            (dir_l and self.is_collision(point_l)[0]) or 
            (dir_u and self.is_collision(point_u)[0]) or 
            (dir_d and self.is_collision(point_d)[0]),
            
            (dir_u and self.is_collision(point_r)[0]) or 
            (dir_d and self.is_collision(point_l)[0]) or 
            (dir_l and self.is_collision(point_u)[0]) or 
            (dir_r and self.is_collision(point_d)[0]),
            
            (dir_d and self.is_collision(point_r)[0]) or 
            (dir_u and self.is_collision(point_l)[0]) or 
            (dir_r and self.is_collision(point_u)[0]) or 
            (dir_l and self.is_collision(point_d)[0]),
            
            # 当前移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 食物位置（相对方向）
            self.food.x < self.head.x,  # 食物在左边
            self.food.x > self.head.x,  # 食物在右边
            self.food.y < self.head.y,  # 食物在上面
            self.food.y > self.head.y,  # 食物在下面
            
            # 新增特征：距离食物的相对距离（归一化）
            (self.food.x - self.head.x) / self.w,  # x方向距离（归一化）
            (self.food.y - self.head.y) / self.h,  # y方向距离（归一化）
            
            # 蛇的长度（归一化）
            len(self.snake) / (self.w * self.h / (BLOCK_SIZE * BLOCK_SIZE)),
            
            # 当前位置到边界的距离（归一化）
            self.head.x / self.w,  # 到左边界的相对距离
            self.head.y / self.h,  # 到上边界的相对距离
        ]
        
        return np.array(state, dtype=float)
        
    def set_speed(self, speed):
        """设置游戏速度"""
        self.speed = speed
        
    def get_snake_length(self):
        """获取蛇的长度"""
        return 3 + self.score  # 正确显示蛇长度为初始长度3加上得分 

    def _calculate_space_utilization_reward(self):
        """计算空间利用效率奖励"""
        total_spaces = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        used_spaces = len(self.snake)
        utilization_rate = used_spaces / total_spaces
        
        # 奖励高效利用空间
        if utilization_rate > 0.3:
            return 2.0  # 高密度奖励
        elif utilization_rate > 0.2:
            return 1.0
        elif utilization_rate > 0.1:
            return 0.5
        else:
            return 0.0 

    def _get_circle_penalty(self):
        """计算转圈惩罚（增强版）"""
        penalty = 0
        
        # 检查是否在原地打转
        if self._is_circle_movement():
            penalty -= 5  # 基础转圈惩罚
        
        # 检查连续相同方向移动（可能卡在角落）
        if self.same_direction_count > 15:
            penalty -= 2
            if self.same_direction_count > 25:
                penalty -= 5  # 严重的单一方向移动惩罚
        
        # 检查位置重复
        recent_positions = self.last_positions[-10:] if len(self.last_positions) >= 10 else self.last_positions
        unique_positions = set(recent_positions)
        
        # 如果最近10步中位置重复度很高
        if len(unique_positions) < len(recent_positions) * 0.6:
            penalty -= 3  # 位置重复惩罚
        
        return penalty
    
    def _calculate_food_direction_reward(self, prev_head):
        """计算朝向食物移动的奖励"""
        # 计算移动前后到食物的向量
        prev_to_food_x = self.food.x - prev_head.x
        prev_to_food_y = self.food.y - prev_head.y
        
        curr_to_food_x = self.food.x - self.head.x
        curr_to_food_y = self.food.y - self.head.y
        
        # 计算移动向量
        move_x = self.head.x - prev_head.x
        move_y = self.head.y - prev_head.y
        
        # 如果移动方向与到食物的方向一致，给予奖励
        reward = 0
        
        # X方向奖励
        if prev_to_food_x > 0 and move_x > 0:  # 食物在右，向右移动
            reward += 0.5
        elif prev_to_food_x < 0 and move_x < 0:  # 食物在左，向左移动
            reward += 0.5
        elif prev_to_food_x > 0 and move_x < 0:  # 食物在右，向左移动
            reward -= 0.3
        elif prev_to_food_x < 0 and move_x > 0:  # 食物在左，向右移动
            reward -= 0.3
        
        # Y方向奖励
        if prev_to_food_y > 0 and move_y > 0:  # 食物在下，向下移动
            reward += 0.5
        elif prev_to_food_y < 0 and move_y < 0:  # 食物在上，向上移动
            reward += 0.5
        elif prev_to_food_y > 0 and move_y < 0:  # 食物在下，向上移动
            reward -= 0.3
        elif prev_to_food_y < 0 and move_y > 0:  # 食物在上，向下移动
            reward -= 0.3
        
        return reward 