import pygame as pg
from keyboard import is_pressed
import random
import time
from math import sqrt
pg.init()




class Game:

    class Player:
        
        SIZE = 18
        COLOR = (252, 239, 61)

        JUMP_STRENGTH = [3, 14]

        def __init__(self, game):
            self.vel = [0, 0]
            self.pos = [game.RESOLUTION[0] // 2, game.TOP_BOUND]
            self.game = game
            self.dead = False
            self.last_jump = False
            self.jump = 0
            self.score = 0
            self.waiting_score_reset = False
        
        def set_jump(self, jump):
            self.jump = jump
 
        def update(self, jump=False, ofset=0, new_pipes=False):
            jump = self.jump
            if self.dead: return ofset
            if new_pipes: self.waiting_score_reset = False
            j = jump
            jump = False if jump == self.last_jump else jump
            self.last_jump = j

            self.pos[1] += ofset

            if jump == 1:
                self.vel = [-self.JUMP_STRENGTH[0], -self.JUMP_STRENGTH[1]]
            if jump == 2:
                self.vel = [self.JUMP_STRENGTH[0], -self.JUMP_STRENGTH[1]]

            self.vel[1] += self.game.GRAVITY

            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]

            if self.pos[0] < 0:
                self.pos[0] = 0
            if self.pos[0] + self.SIZE / 2 > self.game.RESOLUTION[0]:
                self.pos[0] = self.game.RESOLUTION[0] - self.SIZE // 2
            
            pipes = self.game.pipes.lower_pipes_pos
            if self.pos[1]+self.SIZE/2 > pipes[1]-self.game.pipes.PIPE_WIDTH/2 and self.pos[1]-self.SIZE/2 < pipes[1]+self.game.pipes.PIPE_WIDTH/2:
                if (self.pos[0]-self.SIZE/2 < pipes[0]-self.game.pipes.PIPE_HOLE_SIZE/2 or self.pos[0]+self.SIZE/2 > pipes[0]+self.game.pipes.PIPE_HOLE_SIZE/2):
                    self.dead = True
            
            if self.pos[1]+self.SIZE/2 > self.game.RESOLUTION[1]:
                self.dead = True
            
            for block in self.game.blocks.blocks:
                dist = sqrt(pow((self.pos[0]-block[0]), 2)+pow((self.pos[1]-block[1]),2))
                if dist < self.game.blocks.SIZE/2 + self.SIZE/2:
                    self.dead = True

            if not self.dead and not self.waiting_score_reset:
                if self.pos[1] < self.game.pipes.lower_pipes_pos[1]:
                    self.score += 1
                    self.waiting_score_reset = True
            
            ofset = self.game.TOP_BOUND - self.pos[1] if self.pos[1] < self.game.TOP_BOUND else 0

            pg.draw.circle(self.game.screen, self.COLOR, self.pos, self.SIZE // 2)
            return ofset
        
        def get_obstacles(self):
            bellow_blocks = []
            if self.game.blocks.blocks[0][1] > self.pos[1]:
                bellow_blocks = self.game.blocks.blocks[0:2]
            else: bellow_blocks = [[0, self.game.RESOLUTION[1]], self.game.RESOLUTION]
            if bellow_blocks[0][1] > bellow_blocks[1][1]:
                bellow_block_right = bellow_blocks[0]
                bellow_block_left = bellow_blocks[1]
            else:
                bellow_block_right = bellow_blocks[1]
                bellow_block_left = bellow_blocks[0]

            above_blocks = []
            if self.game.blocks.blocks[0][1] < self.game.players[0].pos[1]:
                above_blocks = self.game.blocks.blocks[0:2]
            elif self.game.blocks.blocks[3][1] < self.game.players[0].pos[1]:
                above_blocks = self.game.blocks.blocks[2:4]
            else: above_blocks = [[0, 0], [0, self.game.RESOLUTION[0]]]
            if above_blocks[0][1] > above_blocks[1][1]:
                above_block_right = above_blocks[0]
                above_block_left = above_blocks[1]
            else:
                above_block_right = above_blocks[1]
                above_block_left = above_blocks[0]
            
            if self.game.pipes.lower_pipes_pos[1] < self.pos[1]:
                above_pipes = self.game.pipes.lower_pipes_pos
            else: above_pipes = self.game.pipes.upper_pipes_pos
            
            return bellow_block_right, bellow_block_left, above_block_right, above_block_left, above_pipes

    class Pipes:
        COLOR = (137, 176, 81)
        SIDE_BOUND = 60
        PIPE_WIDTH = 35
        PIPE_HOLE_SIZE = 100
        PIPE_SPACING = 400
        
        def __init__(self, game):
            self.game = game
            self.lower_pipes_pos = [random.randint(self.SIDE_BOUND, self.game.RESOLUTION[0] - self.SIDE_BOUND), -self.PIPE_WIDTH]
            self.upper_pipes_pos = [random.randint(self.SIDE_BOUND, self.game.RESOLUTION[0] - self.SIDE_BOUND), -self.PIPE_WIDTH-self.PIPE_SPACING]

        def _getRects(self, pos):
            leftPipeRect = ((0, pos[1]-self.PIPE_WIDTH//2),
                            (pos[0]-self.PIPE_HOLE_SIZE//2, pos[1]-self.PIPE_WIDTH//2),
                            (pos[0]-self.PIPE_HOLE_SIZE//2, pos[1]+self.PIPE_WIDTH//2),
                            (0, pos[1]+self.PIPE_WIDTH//2))

            rightPipeRect = ((self.game.RESOLUTION[0], pos[1]-self.PIPE_WIDTH//2),
                            (pos[0]+self.PIPE_HOLE_SIZE//2, pos[1]-self.PIPE_WIDTH//2),
                            (pos[0]+self.PIPE_HOLE_SIZE//2, pos[1]+self.PIPE_WIDTH//2),
                            (self.game.RESOLUTION[0], pos[1]+self.PIPE_WIDTH//2))
            
            return leftPipeRect, rightPipeRect

        def update(self, ofset):
            for pipes in [self.lower_pipes_pos, self.upper_pipes_pos]:
                pipes[1] += ofset

                if self.game.HELPER_LINES: pg.draw.circle(self.game.screen, (255, 0, 0), pipes, 2)

                l_rect, r_rect = self._getRects(pipes)
                
                pg.draw.polygon(self.game.screen, self.COLOR, l_rect)
                pg.draw.polygon(self.game.screen, self.COLOR, r_rect)

            if self.lower_pipes_pos[1] > self.game.RESOLUTION[1]+self.PIPE_WIDTH//2:
                self.lower_pipes_pos = self.upper_pipes_pos
                self.upper_pipes_pos = [random.randint(self.SIDE_BOUND, self.game.RESOLUTION[0] - self.SIDE_BOUND), self.lower_pipes_pos[1]-self.PIPE_SPACING]
                return True
            return False

    class Blocks:

        SIZE = 35
        COLOR = (221, 166, 83)
        MIN_BLOCK2BLOCK_DIST = 50

        def __init__(self, game):
            self.game = game
            self.blocks = []
            for l in range(4):
                self.create_blocks(level=l)
            self.new_block_buffer = True

        def create_blocks(self, level=2):
            max_y = int(self.game.pipes.lower_pipes_pos[1] - self.game.pipes.PIPE_WIDTH - self.SIZE - self.game.pipes.PIPE_SPACING * level)
            min_y = int(self.game.pipes.lower_pipes_pos[1] - self.game.pipes.PIPE_SPACING*(level+1) + self.game.pipes.PIPE_WIDTH + self.SIZE)

            x1 = random.randint(self.SIZE, self.game.RESOLUTION[0] - self.SIZE)
            y1 = random.randint(min_y, max_y)
            block = [x1, y1]
            self.blocks.append(block)

            dist = self.MIN_BLOCK2BLOCK_DIST - 1
            while dist < self.MIN_BLOCK2BLOCK_DIST:
                x2 = random.randint(self.SIZE, self.game.RESOLUTION[0] - self.SIZE)
                y2 = random.randint(min_y, max_y)
                dist = sqrt(pow((x2-x1), 2) + pow((y2-y1), 2))

            block = [x2, y2]
            self.blocks.append(block)


        def update(self, ofset, new_blocks=False):

            for block in self.blocks:
                block[1] += ofset
                pg.draw.rect(self.game.screen, self.COLOR, (block[0]-self.SIZE//2, block[1]-self.SIZE//2, self.SIZE, self.SIZE))

            if new_blocks:
                if not self.new_block_buffer:
                    self.create_blocks(2)
                    self.blocks = self.blocks[2:]
                if self.new_block_buffer: self.new_block_buffer = False
            self.blocks.sort(key=lambda x: x[1], reverse=True)

    RESOLUTION = (350, 500)
    FPS =  60
    GRAVITY = 0.8
    BACKGROUND = (0, 136, 146)
    TOP_BOUND = 300
    HELPER_LINES = True
    FONT = pg.font.SysFont('timesnewroman', 20)

    def __init__(self):
        self.screen = pg.display.set_mode(self.RESOLUTION)
        self.clock = pg.time.Clock()
        self.running = False
        self.ofset = 0
    
    def restart(self, players):
        self.high_score = 0
        self.running = True
        self.pipes = self.Pipes(self)
        self.blocks = self.Blocks(self)
        self.players = [self.Player(self) for _ in range(players)]
        self.best_player = self.players[0]

    def update(self):
        
        self.screen.fill(self.BACKGROUND)

        new_pipe = self.pipes.update(self.ofset)
        self.blocks.update(self.ofset, new_blocks=new_pipe)

        if self.HELPER_LINES:
            pg.draw.circle(self.screen, (255, 0, 0), self.blocks.blocks[3], 2)
            pg.draw.circle(self.screen, (255, 0, 0), self.blocks.blocks[2], 2)
            pg.draw.circle(self.screen, (255, 0, 0), self.blocks.blocks[1], 2)
            pg.draw.circle(self.screen, (255, 0, 0), self.blocks.blocks[0], 2)

        score = 0
        for player in self.players:

            max_ofset = 0
            ofset = player.update(ofset=self.ofset, new_pipes=new_pipe)
            if ofset > max_ofset:
                max_ofset = ofset
            self.ofset = max_ofset
        
            if player.score > score: 
                score = player.score
                self.best_player = player

        if score > self.high_score:
            self.high_score = score

        pg.draw.circle(self.screen, (0, 0, 255), self.best_player.pos, self.best_player.SIZE//2, 2)

        if self.HELPER_LINES: 
            pg.draw.line(self.screen, (0, 255, 255), (0, self.TOP_BOUND-self.players[0].SIZE//2), (self.RESOLUTION[0], self.TOP_BOUND-self.players[0].SIZE//2), 1)

    def update_screen(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                pg.quit()
                quit()

        pg.display.flip()
        self.clock.tick(self.FPS)

