import pygame as pg
from keyboard import is_pressed
import random





class Game:

    class Player:
        
        SIZE = 15
        COLOR = (252, 239, 61)

        def __init__(self, game):
            self.vel = [0, 0]
            self.pos = [game.RESOLUTION[0] // 2, game.TOP_BOUND]
            self.game = game

        def update(self, jump=False, ofset=0):

            self.pos[1] += ofset

            if jump == 1:
                self.vel = [-2, -8]
            if jump == 2:
                self.vel = [2, -8]

            self.vel[1] += self.game.GRAVITY

            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]

            if self.pos[0] < 0:
                self.pos[0] = 0
            if self.pos[0] + self.SIZE / 2 > self.game.RESOLUTION[0]:
                self.pos[0] = self.game.RESOLUTION[0] - self.SIZE // 2
            
            ofset = self.game.TOP_BOUND - self.pos[1] if self.pos[1] < self.game.TOP_BOUND else 0

            # for testing
            if self.pos[1] > 1000:
                self.pos[1] = 0
            
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
                pg.draw.circle(self.game.screen, (255, 0, 0), pipes, 1)

                l_rect, r_rect = self._getRects(pipes)
                
                pg.draw.polygon(self.game.screen, self.COLOR, l_rect)
                pg.draw.polygon(self.game.screen, self.COLOR, r_rect)

            if self.lower_pipes_pos[1] > self.game.RESOLUTION[1]+self.PIPE_WIDTH//2:
                self.lower_pipes_pos = self.upper_pipes_pos
                self.upper_pipes_pos = [random.randint(self.SIDE_BOUND, self.game.RESOLUTION[0] - self.SIDE_BOUND), self.lower_pipes_pos[1]-self.PIPE_SPACING]
        


    RESOLUTION = (350, 500)
    FPS =  60
    GRAVITY = 0.5

    BACKGROUND = (0, 136, 146)

    TOP_BOUND = 300

    def __init__(self):
        self.screen = pg.display.set_mode(self.RESOLUTION)
        self.clock = pg.time.Clock()
        self.running = False
        self.ofset = 0

        self.pipes = self.Pipes(self)
    
    def restart(self, players):
        self.running = True
        self.players = [self.Player(self) for _ in range(players)]

    def update(self):
        
        self.screen.fill(self.BACKGROUND)

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
        
        self.pipes.update(self.ofset)
        
        pg.draw.line(self.screen, (255, 0, 0), (0, self.TOP_BOUND-self.players[0].SIZE//2), (self.RESOLUTION[0], self.TOP_BOUND-self.players[0].SIZE//2), 1)

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