import gym
from gym import spaces
import pygame
import numpy as np
import random
import math
import time
from datetime import datetime

BLOCKS_SIZE = 4	
colors = [
    (0, 0, 0)
]

# Figure Class used to generate and rotate figures
class Figure:
    # Initial positions
    x = 0
    y = 0
    
    # Array of figures and orientations
    figures = [
        [[0, 4, 8, 12], [0, 1, 2, 3]],
        [[0, 1, 4, 8], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[0, 1, 5, 9], [4, 5, 6, 8], [0, 4, 8, 9], [2, 4, 5, 6]],  
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [0, 4, 5, 8]],
        [[0, 1, 4, 5]],
        [[0, 1, 5, 6], [1, 5, 4, 8]],
        [[4, 5, 1, 2], [0, 4, 5, 9]]]
    
    '''
    
    '''
    
    # When called with a start x and y
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = 1
        self.rotation = random.randint(0, len(self.figures[self.type]) - 1)
        
    
    # Retrieve a figure    
    def image(self):
        return self.figures[self.type][self.rotation]

    # Rotate a figure
    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])
        
# Class for the Tetris game mechanics
class Tetris:
    level = 2 # Increase Game Speed
    score = 0 
    state = "start" 
    field = [] # Field will be the playing field of ones and zeros
    height = 0
    width = 0
    x = 100
    y = 60
    zoom = 20
    figure = None
    new_set = False
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.field = []
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)
    
    def new_figure(self):
        self.figure = Figure(1, 0)
        self.new_set = True
    
    def intersects(self):
        intersection = False
        
        # Check all 4 possible numbers
        # If the numbers correspond to the shape (Save computational time)
        for i in range(BLOCKS_SIZE):
            for j in range(BLOCKS_SIZE):
                if i * BLOCKS_SIZE + j in self.figure.image(): 
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection
    
    # Confirming Tetromino position and updating the game field values
    # 'new' variable is used when a new tetromino needs to be created
    def freeze(self, new, test):
        for i in range(BLOCKS_SIZE):
            for j in range(BLOCKS_SIZE):
                if i * BLOCKS_SIZE + j in self.figure.image():
                    if not test:
                        self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
                    else:
                        self.field[i + self.figure.y][j + self.figure.x] = 2
                    
        self.break_lines() # Checks to clear any full lines
        if new:
            self.new_figure() # Generates a new figure
        
            if self.intersects():
                self.state = "gameover"
            
    # Checks to clear any full lines
    def break_lines(self):
        lines = 0 # How many lines will break in the movement 
        
        # Check for any spaces
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            
            #If there are no spaces, then break line        
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j] # Move all above lines down 
        # Increase score based on how many lines were cleared
        
        self.score += lines ** 2
    
    # Block Falling Mechanics
    def go_space(self, new):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze(new, False)

    # Observation Space:
    # Saving the field before freezing the falling tetromino
    def save_field(self):
        self.temp_field = []
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.temp_field.append(new_line)
        for i in range(self.height):
            for j in range(self.width):
                self.temp_field[i][j] = self.field[i][j]
        self.temp_x = self.figure.x 
        self.temp_y = self.figure.y
        self.temp_rot = self.figure.rotation
        self.temp_type = self.figure.type 
        self.temp_score = self.score

    # Observation Space:
    # Loading the field after observation space has been set
    def load_field(self):
        for i in range(self.height):
            for j in range(self.width):
                self.field[i][j] = self.temp_field[i][j]
        self.figure.x = self.temp_x
        self.figure.y = self.temp_y
        self.figure.rotation = self.temp_rot
        self.figure.type = self.temp_type
        self.score = self.temp_score
    
    # Block falls in increments
    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze(True, False)
    
    #Block move left or right based on dx (-1 for left and 1 for right)
    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x
    
    # The rotation process asigns the next rotation in the figure class
    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation       
            
# Parameters for the PyGame Render and Action Space
N_DISCRETE_ACTIONS = 40

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Gym Environment Implimentation
class BlocksEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    rendering = False
    save_files = False
    high_score = 0
    total_score = 0
    games = 1
    file_path = "C:/Users/Divan van der Bank/OneDrive - Stellenbosch University/Divan Ingenieurswese 4de Jaar/Skripsie/Rl_Tetris_V1.0/Melax.txt"
    test2_s = False
    obsFlatten = False
    total_reward = 0
    test_obs = False
    obs_space = 15

    def __init__(self):
        super(BlocksEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        if self.obsFlatten:
            self.observation_space = spaces.Box(low=np.zeros(200), high=np.ones(200), shape=(200,), dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=np.zeros([20,10]), high=np.ones([20,10]), shape=(20,10), dtype=np.float64)

    def step(self, action):
        if self.game.figure is None:
            self.game.new_figure()

        
        x_pos = action % 10
        self.game.figure.rotation = np.min([len(Figure.figures[self.game.figure.type]) - 1, math.floor(action / 10)])
        
        prev_x = 0
        while prev_x != self.game.figure.x:
            prev_x = self.game.figure.x 
            if x_pos < self.game.figure.x:
                self.game.go_side(-1)
            elif x_pos > self.game.figure.x:
                self.game.go_side(1)
    
        self.game.go_space(True)

        if self.rendering:
                pygame.event.get()
                self.render()

        if (self.game.state == 'gameover'):
            self.total_score += self.game.score	
            info = {"r": self.total_reward, "l": self.counter, "d" : True}		
            
            if self.high_score > 0:
                if self.save_files:
                    file = open(self.file_path, "a")
                self.games += 1
            else:
                if self.save_files:
                    file = open(self.file_path, "w")
                    file.write(f"Game Nr\tGame Score\n")
            if self.game.score > self.high_score:
                self.high_score = self.game.score
            if self.save_files:
                file.write(f"{self.games}\t{self.game.score}\n")
                file.close()
            self.done = True
        else:
            info = {"r": 0, "l": 0, "d" : False}	

        


        self.columns_height, self.reward = self.values(np.array(self.game.field), self.done)
        self.total_reward += self.reward

        if not self.test_obs:
            
            self.game.save_field()
            self.game.freeze(False, self.test2_s)
            self.observation = np.array(self.game.field)
            # print(self.observation)
            if self.obsFlatten:
                self.observation = self.observation.flatten()
            
            self.game.load_field()
        else:
            fig_type = np.zeros(7)
            fig_type[self.game.figure.type] = 1
        
            # col_height = np.zeros([17,10])
            # for i in range(10):
            #     if self.columns_height[i] >= 17:
            #         col_height[16][i] = 1
            #     else:
            #         col_height[self.columns_height[i]][i] = 1

            # col_height = col_height.flatten()
            
            self.observation = list(self.columns_height) + list(fig_type)# [self.game.figure.type]
            self.observation = np.array(self.observation)
        extra = {}
        

        return self.observation, self.reward, self.done, info, extra

    def reset(self):		
        
        size = (600, 500)
        if self.rendering:
            pygame.init()
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Block Falls")
            self.clock = pygame.time.Clock()	
            self.fps = 25

        self.best_coord = [0,0]

        self.done = False
        self.game = Tetris(20, 10)
        for i in range(self.game.height):
            for j in range(self.game.width):
                self.game.field[i][j] = 0

        self.prev_reward = 0
        self.cleared = 0
        self.prevSTD = 0
        self.prevHoles = 0
        self.counter = 0
        self.total_reward = 0
        self.prevHeight = 0

        self.columns_height = np.zeros(self.game.width, dtype=int)
        self.row_nr = np.zeros(self.game.height)

        if not self.test_obs:
            self.observation = np.array(self.game.field)
            if self.obsFlatten:
                self.observation = self.observation.flatten()
        else:
            self.observation = np.zeros(self.obs_space)

        return self.observation

    def render(self, mode='human'):
        # Initialize the game engine
        self.screen.fill(WHITE) # Keep the screen white
    
        # Draw the screen of rectangles
        for i in range(self.game.height):
            for j in range(self.game.width):
                pygame.draw.rect(self.screen, GRAY, [50 + self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i, self.game.zoom,self. game.zoom], 1)
                if self.game.field[i][j] > 0:
                    pygame.draw.rect(self.screen, 1,
                                    [50 + self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1, self.game.zoom - 2, self.game.zoom - 1])

        if self.game.figure is not None:
            for i in range(BLOCKS_SIZE):
                for j in range(BLOCKS_SIZE):
                    p = i * BLOCKS_SIZE + j
                    if p in self.game.figure.image():
                        pygame.draw.rect(self.screen, 0,
                                        [50 + self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                        self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                        self.game.zoom - 2, self.game.zoom - 2])

        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(self.game.score) + "  -  High Score: " + str(self.high_score) + " - Average: " + str(round(self.total_score / self.games)) + " - Games: " + str(self.games), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        self.screen.blit(text, [0, 0])
        if self.game.state == "gameover":
            self.screen.blit(text_game_over, [20, 200])
            self.screen.blit(text_game_over1, [25, 265])
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.display.quit()
        pygame.quit()
            

    def values(self, arrGameField, gameover):
        nr_holes = 0

        columns_height = np.zeros(self.game.width, dtype=int)
        for i in range(arrGameField.shape[1]):
            first = True
            for j in range(arrGameField.shape[0]):				
                if (arrGameField[j][i]== 1) & (first == True):
                    columns_height[i] = arrGameField.shape[0] - j
                    first = False
                if (arrGameField[j][i] == 0) & (first == False):
                    nr_holes += 1

        if (np.max(columns_height) >= 17): # arrGameField.shape[0] - 1
            self.game.state = 'gameover'

        stdDev = np.std(columns_height)

        if (self.prev_reward < self.game.score):
            self.cleared = self.game.score - self.prev_reward
        else:
            self.cleared = 0
        self.prev_reward = self.game.score

        if gameover:
            game_reward = -1
        else :
            game_reward =  float((self.prevHeight - np.max(columns_height)) + (self.prevSTD - stdDev) + (self.prevHoles - nr_holes) / 2)
        
        self.prevHoles = nr_holes
        self.prevSTD = stdDev
        self.prevHeight = np.max(columns_height)
    
        return columns_height, game_reward
