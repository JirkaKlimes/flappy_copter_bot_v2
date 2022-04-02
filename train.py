from flappy_copter import Game
from deep_learning import Population, Layer, Agent
import pygame as pg
from keyboard import is_pressed

game = Game()
game.restart(1)

while True:

    game.update()

    obstacles = game.best_player.get_obstacles()
    bellow_block_right = obstacles[0]
    bellow_block_left = obstacles[1]
    above_block_right = obstacles[2]
    above_block_left = obstacles[3]
    above_pipes = obstacles[4]

    pg.draw.aaline(game.screen, (0, 0, 255), above_block_right, game.best_player.pos)
    pg.draw.aaline(game.screen, (0, 0, 255), above_block_left, game.best_player.pos)
    pg.draw.aaline(game.screen, (255, 0, 0), bellow_block_right, game.best_player.pos)
    pg.draw.aaline(game.screen, (255, 0, 0), bellow_block_left, game.best_player.pos)

    pg.draw.aaline(game.screen, (0, 255, 0), above_pipes, game.best_player.pos)

    for player in game.players:
        obstacles = player.get_obstacles()
        bellow_block_right = obstacles[0]
        bellow_block_left = obstacles[1]
        above_block_right = obstacles[2]
        above_block_left = obstacles[3]
        above_pipes = obstacles[4]

        playing = False
        jump = 0
        if is_pressed('a'):
            jump = 1
        if is_pressed('d'):
            jump = 2
        if not player.dead:
            playing = True
        player.set_jump(jump)
        if not playing:
            game.restart(1)
            continue

    game.update_screen()