import pygame as pg
from keyboard import is_pressed
import random
import time




class Game:

    class Player:
        
        SIZE = 18
        COLOR = (252, 239, 61)

        JUMP_STRENGTH = [3, 18]

        def __init__(self, game):
            self.vel = [0, 0]
            self.pos = [game.RESOLUTION[0] // 2, game.TOP_BOUND]
            self.game = game
            self.dead = False
            self.last_jump = False

        def update(self, jump=False, ofset=0):
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
            
            ofset = self.game.TOP_BOUND - self.pos[1] if self.pos[1] < self.game.TOP_BOUND else 0

            pg.draw.circle(self.game.screen, self.COLOR, self.pos, self.SIZE // 2)

            return ofset

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

    class Blocks:

        SIZE = 35
        COLOR = (221, 166, 83)

        def __init__(self, game):
            self.game = game

        def update(self, ofset):
            pass

    RESOLUTION = (350, 500)
    FPS =  60
    GRAVITY = 1.3
    BACKGROUND = (0, 136, 146)
    TOP_BOUND = 300
    HELPER_LINES = True

    def __init__(self):
        self.screen = pg.display.set_mode(self.RESOLUTION)
        self.clock = pg.time.Clock()
        self.running = False
        self.ofset = 0
    
    def restart(self, players):
        self.running = True
        self.pipes = self.Pipes(self)
        self.blocks = self.Blocks(self)
        self.players = [self.Player(self) for _ in range(players)]

    def update(self):
        
        self.screen.fill(self.BACKGROUND)

        self.pipes.update(self.ofset)
        self.blocks.update(self.ofset)

        for player in self.players:
            jump = 0
            if is_pressed('a'):
                jump = 1
            if is_pressed('d'):
                jump = 2

            max_ofset = 0
            ofset = player.update(jump, self.ofset)
            if ofset > max_ofset:
                max_ofset = ofset
            self.ofset = max_ofset

            if player.dead: self.restart(1)
        
        if self.HELPER_LINES: pg.draw.line(self.screen, (255, 0, 0), (0, self.TOP_BOUND-self.players[0].SIZE//2), (self.RESOLUTION[0], self.TOP_BOUND-self.players[0].SIZE//2), 1)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                pg.quit()
                quit()

        pg.display.flip()
        self.clock.tick(self.FPS)

game = Game()

game.restart(1)

while True:
    game.update()