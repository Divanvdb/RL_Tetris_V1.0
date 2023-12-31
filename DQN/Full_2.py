import gym
from gym import spaces
import pygame
import numpy as np
import random
import time
from datetime import datetime

BLOCKS_SIZE = 4
# Set the colors for the game. The colors will be chosen from this array

colors = [
    (0, 0, 0)
]

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
    
class Tetris:
    level = 2
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
        for i in range(BLOCKS_SIZE):
            for j in range(BLOCKS_SIZE):
                if i * BLOCKS_SIZE + j in self.figure.image(): # If the numbers correspond to the shape (Save computational time)
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection
    
    # Setting the field values to the corresponding color
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
    
    # For the block to fall
    def go_space(self, new):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze(new, False)

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

N_DISCRETE_ACTIONS = 4


# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)




class BlocksEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    rendering = False
    save_files = False
    high_score = 0
    total_score = 0
    games = 1
    file_path = ""
    test2_s = False
    obsFlatten = False
    total_reward = 0

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
            
        # Increase counter and overflow
        self.counter += 1
        if self.counter > 100000:
            self.counter = 0
        
        # Move the Figure down when the counter timer interval triggers or down-arrow is pressed
        if ((self.game.state == "start") & (self.counter % 4 == 0)):
            self.game.go_down()
        
        #Events in the game
        if self.rendering:
            pygame.event.get()
        if action >= 0: # When any key on the keyboard is pressed
            if action == 0: # When the UP arrow is pressed, rotate
                self.game.rotate()
            if action == 1: # When LEFT arrow move left
                self.game.go_side(-1)
            if action == 2: # When RIGHT arrow move right 
                self.game.go_side(1)
            if action == 3: # When SPACE the block falls
                self.game.go_space(True)

        if self.game.figure.y > 2:
            self.game.go_space(True)

        if (self.game.state == 'gameover'):
            self.total_score += self.game.score	
            # info = {"r": self.total_reward, "l": self.counter, "d" : True}		
            
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
        # else:
        #     info = {"r": 0, "l": 0, "d" : False}	


        # Will render the moves of the RL Agent
        if self.rendering:
            self.render()           
            
        
        self.columns_height, holes, self.reward, _ = self.values(np.array(self.game.field), self.done, self.counter)
        self.total_reward += self.reward
        # self.observation = list(self.columns_height) + list(fig_type) + list(fig_rot) + [self.game.figure.x, self.game.figure.y]	
        self.game.save_field()
        self.game.freeze(False, self.test2_s)
        self.observation = np.array(self.game.field)
        if self.obsFlatten:
            self.observation = self.observation.flatten()
        self.game.load_field()

        info = {}

        return self.observation, self.reward, self.done, info

    def reset(self):
        if self.rendering:
            size = (600, 500)
            pygame.init()
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Block Falls")
            self.clock = pygame.time.Clock()		
            self.fps = 25

        self.best_coord = [0,0]
        self.info = 0
        self.done = False
        self.game = Tetris(20, 10)
        for i in range(self.game.height):
            for j in range(self.game.width):
                self.game.field[i][j] = 0
        self.counter = 0
        self.greedy = False

        self.prevSTD = 0
        self.prevHoles = 0

        self.prev_reward = 0
        self.prev_x = 0
        self.cleared = 0

        self.columns_height = np.zeros(self.game.width, dtype=int)

        #self.observation = list(self.columns_height) + [0] + [0, 0] + [0, 0] + [0 ,0]

        self.observation = np.array(self.game.field)
        if self.obsFlatten:
            self.observation = self.observation.flatten()

        return self.observation
    
    def rule(self):
        if self.game.figure is None:
            self.game.new_figure()
        max_reward = -10
        temp_prev_reward = self.prev_reward
        if self.game.new_set:
            self.game.new_set = False
                
            self.best_coord  = [0, 0]
            self.game.save_field()
            for i in range(self.game.width):
                for j in range(len(self.game.figure.figures[self.game.figure.type])):
                    try:
                        self.game.figure.x = i
                        self.game.figure.rotation = j
                        self.game.go_space(False)
                        #self.render()
                        zz, xx,rr, temp_reward = self.values(np.array(self.game.field), self.done, self.counter)
                        if temp_reward > max_reward:
                            self.best_coord  = [i, j]
                            max_reward = temp_reward
                    except:
                        xx = 0
                    self.game.load_field()
                    self.prev_reward = temp_prev_reward
            self.game.load_field()
        
        # 0 - Rotate
        # 1 - Right
        # 2 - left
        # 3 - Drop
            
        # if ((self.info != 0) & (self.prev_x == self.game.figure.x)):
        #      self.info = 0
        # else:
        self.info = 3
        
        if(self.best_coord[0] < self.game.figure.x):
             self.info = 1
        if(self.best_coord[0] > self.game.figure.x):
             self.info = 2
        if(self.best_coord[1] != self.game.figure.rotation):
            self.info = 0     

        
        # self.prev_x = self.game.figure.x
        return self.info


    def render(self, mode='human'):
        	

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
            

    def values(self, arrGameField, gameover, counts):
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
                    

        low_col = np.zeros(self.game.width, dtype=int)
        low_col[np.where(columns_height == np.min(columns_height))] = 1
            # Number of Holes
        
            # Standard Deviation
        stdDev = np.std(columns_height)

        
        if (self.prev_reward < self.game.score):
            self.cleared = self.game.score - self.prev_reward
        else:
            self.cleared = 0
        self.prev_reward = self.game.score

        best_reward = 0
        if gameover:
            game_reward = -10
        else :
            game_reward = float(self.cleared * 1 + (self.prevSTD- stdDev) - (nr_holes - self.prevHoles)/10)# + 0.01 #- 0.005 * stdDev - 0.001 * np.sum(columns_height) + (10 - np.max(columns_height))/100
            best_reward = game_reward * 0.8  - stdDev * 0.4 - 0.8 * nr_holes
        
        self.prevSTD = stdDev
        self.prevHoles = nr_holes
    
        return columns_height, nr_holes, game_reward, best_reward

        

