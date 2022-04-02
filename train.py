from flappy_copter import Game
from deep_learning import Population, Layer, Agent, NeuralNetwork
import pygame as pg
from keyboard import is_pressed
import numpy as np

n_players = 100

rate = 0.2
scale = 0.04
pool_size = 4
max_updates = 200
method = Population.TOP_X
include_parents = True
mutated_layers = NeuralNetwork.ALL

# layer_0 = Layer(12, 16, activation=Layer.RELU)
# layer_1 = Layer(16, 16, activation=Layer.RELU)
# layer_2 = Layer(16, 8, activation=Layer.RELU)
# layer_3 = Layer(8, 8, activation=Layer.RELU)
# layer_4 = Layer(8, 3, activation=Layer.SOFTMAX)
# agent = Agent([layer_0, layer_1, layer_2, layer_3, layer_4])
# population = Population(agent, n_players)

population = Population(file_name='population.npy')

population.evolution_settings(include_parents=include_parents, 
                              pool_size=pool_size, selection_method=method, 
                              population_size=n_players)

population.mutation_settings(mutation_rate=rate, mutation_scale=scale, mutated_layers=mutated_layers)






game = Game()

gen = 0
while True:
    game.restart(n_players)
    gen += 1
    playing = True
    updates = 0
    max_score = 0
    while playing and updates < max_updates:
        updates += 1
        if game.best_player.score > max_score:
            max_score = game.best_player.score
            updates = 0
        game.update()

        obstacles = game.best_player.get_obstacles()
        block_dr = obstacles[0]
        block_dl = obstacles[1]
        block_ur = obstacles[2]
        block_ul = obstacles[3]
        pipes = obstacles[4]

        pg.draw.aaline(game.screen, (0, 0, 255), block_ur, game.best_player.pos)
        pg.draw.aaline(game.screen, (0, 0, 255), block_ul, game.best_player.pos)
        pg.draw.aaline(game.screen, (255, 0, 0), block_dr, game.best_player.pos)
        pg.draw.aaline(game.screen, (255, 0, 0), block_dl, game.best_player.pos)

        pg.draw.aaline(game.screen, (0, 255, 0), pipes, game.best_player.pos)

        some1_alive = False
        for player, agent in zip(game.players, population.agents):
            if not player.dead: some1_alive = True
            obstacles = player.get_obstacles()
            block_dr = obstacles[0]
            block_dl = obstacles[1]
            block_ur = obstacles[2]
            block_ul = obstacles[3]
            pipes = obstacles[4]

            block_dr_x = block_dr[0]-player.pos[0]/game.RESOLUTION[0]
            block_dr_y = block_dr[1]-player.pos[1]/game.RESOLUTION[1]
            block_dl_x = block_dl[0]-player.pos[0]/game.RESOLUTION[0]
            block_dl_y = block_dl[1]-player.pos[1]/game.RESOLUTION[1]
            block_ur_x = block_ur[0]-player.pos[0]/game.RESOLUTION[0]
            block_ur_y = block_ur[1]-player.pos[1]/game.RESOLUTION[1]
            block_ul_x = block_ul[0]-player.pos[0]/game.RESOLUTION[0]
            block_ul_y = block_ul[1]-player.pos[1]/game.RESOLUTION[1]
            pipes_x = pipes[0]-player.pos[0]/game.RESOLUTION[0]
            pipes_y = pipes[1]-player.pos[1]/game.RESOLUTION[1]
            vel_x = player.vel[0]/game.RESOLUTION[0]
            vel_y = player.vel[1]/game.RESOLUTION[1]

            inputs = [block_dr_x, block_dr_y, block_dl_x, block_dl_y, block_ur_x, block_ur_y, block_ul_x, block_ul_y, pipes_x, pipes_y, vel_x, vel_y]

            agent.push_forward(inputs)
            jump = np.argmax(agent.outputs)

            agent.fitness = player.score*100.00001 + player.updates

            # jump = 0
            # if is_pressed('a'):
            #     jump = 1
            # if is_pressed('d'):
            #     jump = 2

            player.set_jump(jump)
        playing = some1_alive

        game.update_screen()
    population.save(f'population.npy', rewrite=True)
    population.evolve()