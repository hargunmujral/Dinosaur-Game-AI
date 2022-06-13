import pygame
import os
import sys
import random
import neat

pygame.init()

# Constants

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUN = [pygame.image.load(os.path.join("imports/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("imports/Dino", "DinoRun2.png")),]

JUMP = pygame.image.load(os.path.join("imports/Dino", "DinoJump.png"))

SMALL_CACTUS = [pygame.image.load(os.path.join("imports/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("imports/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("imports/Cactus", "SmallCactus3.png")),]

LARGE_CACTUS = [pygame.image.load(os.path.join("imports/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("imports/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("imports/Cactus", "LargeCactus3.png")),]


BACKGROUND = pygame.image.load(os.path.join("imports/Other", "Track.png"))

FONT = pygame.font.SysFont("freesansbold.ttf", 20)

# Object classes

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    JUMP_VELOCITY = 8.5

    def __init__(self, img=RUN[0]):
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        self.jump_velocity = self.JUMP_VELOCITY
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.step_index = 0

    def update(self):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.step_index >= 10:
            self.step_index = 0

    def jump(self):
        self.image = JUMP
        if self.dino_jump:
            self.rect.y -= self.jump_velocity * 4
            self.jump_velocity -= 0.8
        if self.jump_velocity <= -self.JUMP_VELOCITY:
            self.dino_jump = False
            self.dino_run = True
            self.jump_velocity = self.JUMP_VELOCITY

    def run(self):
        self.image = RUN[self.step_index // 5]
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index += 1
    
    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))

class Obstacle:
    def __init__(self, img, number_of_cacti):
        self.image = img
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()
    
    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, img, number_of_cacti):
        super().__init__(img, number_of_cacti)
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, img, number_of_cacti):
        super().__init__(img, number_of_cacti)
        self.rect.y = 300

# Function Definitions

def distance(pos_a, pos_b):
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return (dx**2 + dy**2)**0.5

def remove(index):
    dinos.pop(index)
    genomes_list.pop(index)
    nets.pop(index)

# Main function

def eval_genomes(genomes, config):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles, dinos, genomes_list, nets
    clock = pygame.time.Clock()
    points = 0

    obstacles = []
    dinos = []
    genomes_list = []
    nets = []

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

    for genome_id, genome in genomes:
        dinos.append(Dinosaur())
        genomes_list.append(genome)
        nets.append( neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0

    def score():
        global points, game_speed
        points += 1
        if points % 10 == 0:
            game_speed += 1
        score = FONT.render(f'Points: {(str(points))}', True, (0, 0, 0))
        SCREEN.blit(score, (950, 50))

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BACKGROUND.get_width()
        SCREEN.blit(BACKGROUND, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BACKGROUND, (x_pos_bg + image_width, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    # Run loop

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))

        for dino in dinos:
            dino.update()
            dino.draw(SCREEN)

        if len(dinos) == 0:
            break

        if len(obstacles) == 0:
            rand_int = random.randint(0, 1)
            if rand_int == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))
            else:
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))
            
        for obstacle in obstacles:
            obstacle.update()
            obstacle.draw(SCREEN)
            for i, dino in enumerate(dinos):
                if dino.rect.colliderect(obstacle.rect):
                    genomes_list[i].fitness -= 1
                    remove(i)

        for i, dino in enumerate(dinos):
            output = nets[i].activate((
                dino.rect.y, 
                distance((dino.rect.x, dino.rect.y), 
                obstacle.rect.midtop)))
            if output[0] > 0.5 and dino.rect.y == dino.Y_POS:
                dino.dino_jump = True
                dino.dino_run = False
        score()
        background()
        clock.tick(30)
        pygame.display.update()

# Setting up NEAT

def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)
    pop.run(eval_genomes, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)