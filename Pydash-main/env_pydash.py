import os
import random
import csv

import gym
import numpy as np
import pygame
from gym import spaces
import sys
import pydash


class PydashEnv():
    
    # Define constants for clearer code
    NO_JUMP = 0
    JUMP = 1

    def __init__(self, size=(600, 400), gravity=0.25, frame_rate=60, render=False):
        super(PydashEnv, self)
        self.FRAME_RATE = frame_rate
        self.player_velocity = 0
        self.step_size = 4
        self.fill = 0
        self.num = 0
        self.CameraX = 0
        self.attempts = 0
        self.coins = 0
        self.angle = 0
        self.level = 0

        # list
        self.particles = []
        self.orbs = []
        self.win_cubes = []

        pygame.init()
        # creates a screen variable of size 800 x 600
        self.screen = pygame.display.set_mode([800, 600])
        # sets the frame rate of the program
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("lucidaconsole", 20)

        # square block face is main character the icon of the window is the block face
        self.avatar = pygame.image.load(os.path.join("images", "avatar.png"))  # load the main character
        pygame.display.set_icon(self.avatar)
        #  this surface has an alpha value with the colors, so the player trail will fade away using opacity
        self.alpha_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # sprite groups
        self.player_sprite = pygame.sprite.Group()
        self.elements = pygame.sprite.Group()

        # images
        self.spike = pygame.image.load(os.path.join("images", "obj-spike.png"))
        self.spike = pydash.resize(self.spike)
        self.coin = pygame.image.load(os.path.join("images", "coin.png"))
        self.coin = pygame.transform.smoothscale(self.coin, (32, 32))
        self.block = pygame.image.load(os.path.join("images", "block_1.png"))
        self.block = pygame.transform.smoothscale(self.block, (32, 32))
        self.orb = pygame.image.load((os.path.join("images", "orb-yellow.png")))
        self.orb = pygame.transform.smoothscale(self.orb, (32, 32))
        self.trick = pygame.image.load((os.path.join("images", "obj-breakable.png")))
        self.trick = pygame.transform.smoothscale(self.trick, (32, 32))

        # initialize level with
        self.levels = ["level_1.csv"]
        self.level_list = self.block_map(self.levels[0])
        self.level_width = (len(self.level_list[0]) * 32)
        self.level_height = len(self.level_list) * 32
        self.init_level(self.level_list)

        # set window title suitable for game
        pygame.display.set_caption('Pydash: Geometry Dash in Python')

        # initialize the font variable to draw text later
        self.text = self.font.render('image', False, (255, 255, 0))

        # music
        #music = pygame.mixer_music.load(os.path.join("music", "bossfight-Vextron.mp3"))
        #pygame.mixer_music.play()

        # bg image
        self.bg = pygame.image.load(os.path.join("images", "bg.png"))

        # create object of player class
        self.player = pydash.Player(self.avatar, self.elements, (150, 150), self.player_sprite)


        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
    
        self.observation_space = spaces.Box(low=0, high=800,
                                            shape=(1,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        """resets the sprite groups, music, etc. for death and new level"""
        #global player, elements, player_sprite, level
        self.player_velocity = 0

        self.player_sprite = pygame.sprite.Group()
        self.elements = pygame.sprite.Group()
        self.player = pydash.Player(self.avatar, self.elements, (150, 150), self.player_sprite)
        pydash.init_level(
                pydash.block_map(
                        level_num=self.levels[self.level]))
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.player_velocity]).astype(np.float32)
    
    def sample_action(self):
        return random.choice([0,1])
    
    """
    Functions
    """
    def block_map(self,level_num):
        """
        :type level_num: rect(screen, BLACK, (0, 0, 32, 32))
        open a csv file that contains the right level map
        """
        lvl = []
        with open(level_num, newline='') as csvfile:
            trash = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in trash:
                lvl.append(row)
        return lvl

    def init_level(self, map):
        """this is similar to 2d lists. it goes through a list of lists, and creates instances of certain obstacles
        depending on the item in the list"""
        x = 0
        y = 0

        for row in map:
            for col in row:

                if col == "0":
                    pydash.Platform(self.block, (x, y), self.elements)

                if col == "Coin":
                    pydash.Coin(self.coin, (x, y), self.elements)

                if col == "Spike":
                    pydash.Spike(self.spike, (x, y), self.elements)
                if col == "Orb":
                    self.orbs.append([x, y])

                    pydash.Orb(self.orb, (x, y), self.elements)

                if col == "T":
                    pydash.Trick(self.trick, (x, y), self.elements)

                if col == "End":
                    pydash.End(self.avatar, (x, y), self.elements)
                x += 32
            y += 32
            x = 0

    def blitRotate(surf, image, pos, originpos: tuple, angle: float):
        """
        rotate the player
        :param surf: Surface
        :param image: image to rotate
        :param pos: position of image
        :param originpos: x, y of the origin to rotate about
        :param angle: angle to rotate
        """
        # calcaulate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pydash.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]

        # make sure the player does not overlap, uses a few lambda functions(new things that we did not learn about number1)
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
        # calculate the translation of the pivot
        pivot = pydash.Vector2(originpos[0], -originpos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (pos[0] - originpos[0] + min_box[0] - pivot_move[0], pos[1] - originpos[1] - max_box[1] + pivot_move[1])

        # get a rotated image
        rotated_image = pygame.transform.rotozoom(image, angle, 1)

        # rotate and blit the image
        surf.blit(rotated_image, origin)

    def eval_outcome(self, won: bool, died: bool):
        """simple function to run the win or die screen after checking won or died"""
        done = False
        if won:
            #won_screen()
            done = True
            self.reset()

        if died:
            #death_screen()
            done = True
            self.reset()
        return done

    def move_map(self):
        """moves obstacles along the screen"""
        for sprite in self.elements:
            sprite.rect.x -= self.CameraX

    def draw_stats(self,surf, money=0):
        """
        draws progress bar for level, number of attempts, displays coins collected, and progressively changes progress bar
        colors
        """
        progress_colors = [pygame.Color("red"), pygame.Color("orange"), pygame.Color("yellow"), pygame.Color("lightgreen"),
                        pygame.Color("green")]

        tries = self.font.render(f" Attempt {str(self.attempts)}", True, pydash.WHITE)
        BAR_LENGTH = 600
        BAR_HEIGHT = 10
        for i in range(1, money):
            self.screen.blit(self.coin, (BAR_LENGTH, 25))
        self.fill += 0.5
        outline_rect = pygame.Rect(0, 0, BAR_LENGTH, BAR_HEIGHT)
        fill_rect = pygame.Rect(0, 0, self.fill, BAR_HEIGHT)
        col = progress_colors[int(self.fill / 100)]
        pydash.rect(surf, col, fill_rect, 0, 4)
        pydash.rect(surf, pydash.WHITE, outline_rect, 3, 4)
        surf.blit(tries, (BAR_LENGTH, 0))

    def move_map(self):
        """moves obstacles along the screen"""
        for self.sprite in self.elements:
            self.sprite.rect.x -= self.CameraX

    def Play(self):
        
        self.player.vel.x = 6

        self.alpha_surf.fill((255, 255, 255, 1), special_flags=pygame.BLEND_RGBA_MULT)

        self.player_sprite.update()
        self.CameraX = self.player.vel.x  # for moving obstacles
        self.move_map()  # apply CameraX to all elements

        self.screen.blit(self.bg, (0, 0))  # Clear the screen(with the bg)

        self.player.draw_particle_trail(self.player.rect.left - 1, self.player.rect.bottom + 2,
                                pydash.WHITE)
        self.screen.blit(self.alpha_surf, (0, 0))  # Blit the alpha_surf onto the screen.
        self.draw_stats(self.screen, pydash.coin_count(self.coins))

        if self.player.isjump:
            """rotate the player by an angle and blit it if player is jumping"""
            pydash.angle -= 8.1712  # this may be the angle needed to do a 360 deg turn in the length covered in one jump by player
            self.blitRotate(self.screen, self.player.image, self.player.rect.center, (16, 16), pydash.angle)
        else:
            """if player.isjump is false, then just blit it normally(by using Group().draw() for sprites"""
            self.player_sprite.draw(self.screen)  # draw player sprite group
        self.elements.draw(self.screen)  # draw all other obstacles

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    """User friendly exit"""
                    pygame.quit()
                    sys.exit()

        pygame.display.flip()
        self.clock.tick(self.FRAME_RATE)

    def step(self, action):
        print("in step function")
        self.Play()
        if action == self.JUMP:
            self.jump_velocity = self.player.jump_amount
            self.player.jump()
        elif action == self.NO_JUMP:
            pass
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        #self.Play()
        # Did we win or lose the game?
        done = self.eval_outcome(self.player.won, self.player.died)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 5
        
        return np.array(self.jump_velocity, reward, done)

    def render(self):
        pass

    def close(self):
        pass
    
#env = PydashEnv()
#while True:
 #   env.Play()