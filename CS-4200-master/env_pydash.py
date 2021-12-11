import os
import random
import csv
import cv2

import numpy as np
import pygame
import sys
import pydash


class PydashEnv:
    # Define constants for clearer code
    NO_JUMP = 0
    JUMP = 1

    def __init__(self, size=(800, 600), gravity=0.25, frame_rate=60, render=True):
        super(PydashEnv, self)
        self.observation_shape = 14400
        #self.observation_shape = 2  # For using standard metrics
        self.action_shape = 2
        self.FRAME_RATE = frame_rate
        self.player_pos = 0
        self.step_size = 4
        self.fill = 0
        self.num = 0
        self.CameraX = 0
        self.attempts = 0
        self.coins = 0
        self.angle = 0
        self.level = 0
        self.player_grid_pos = 0
        self.next_spike_distance = 0
        self.i = 0

        self.render = render

        # list
        self.particles = []
        self.orbs = []
        self.win_cubes = []

        # square block face is main character the icon of the window is the block face
        self.avatar = pygame.image.load(os.path.join("images", "avatar.png"))  # load the main character

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


        if self.render:
            pygame.init()
            # creates a screen variable of size 800 x 600
            self.screen = pygame.display.set_mode([800, 600])
            # sets the frame rate of the program
            self.clock = pygame.time.Clock()

            self.font = pygame.font.SysFont("lucidaconsole", 20)
            
            pygame.display.set_icon(self.avatar)
            #  this surface has an alpha value with the colors, so the player trail will fade away using opacity
            self.alpha_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            
            # set window title suitable for game
            pygame.display.set_caption('Pydash: Geometry Dash in Python')

            # initialize the font variable to draw text later
            self.text = self.font.render('image', False, (255, 255, 0))

            # music
            # music = pygame.mixer_music.load(os.path.join("music", "bossfight-Vextron.mp3"))
            # pygame.mixer_music.play()

            # bg image
            self.bg = pygame.image.load(os.path.join("images", "bg.png"))

        # create object of player class
        self.player = pydash.Player(self.avatar, self.elements, (150, 150), self.player_sprite)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        """resets the sprite groups, music, etc. for death and new level"""
        # global player, elements, player_sprite, level
        self.player_pos = 0
        self.fill = 0
        self.player_sprite = pygame.sprite.Group()
        self.elements = pygame.sprite.Group()
        self.player = pydash.Player(self.avatar, self.elements, (150, 150), self.player_sprite)
        self.init_level(
            self.block_map(
                level_num=self.levels[self.level]))
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        #return np.array([self.fill]).astype(np.float32)
        return self.get_observation()

    def get_observation(self):
        observation = [self.player_grid_pos + random.random(), self.next_spike_distance + random.random()]

        # dim = (120,120)
        pygame.image.save(self.screen, "screenshots/screenshot"+str(self.i)+".jpeg")
        self.i += 1
        # img = cv2.imread("screenshot.jpeg")
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # cv2.imwrite("resizedimg.jpeg",img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("greyscale.jpeg",img)
        # observation = np.asarray(img).ravel()

        return observation
    
    def sample_action(self):
        return random.choice([0, 1])

    """
    Functions
    """

    def block_map(self, level_num):
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

    def blitRotate(self, surf, image, pos, originpos: tuple, angle: float):
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
        origin = (
        pos[0] - originpos[0] + min_box[0] - pivot_move[0], pos[1] - originpos[1] - max_box[1] + pivot_move[1])

        # get a rotated image
        rotated_image = pygame.transform.rotozoom(image, angle, 1)

        # rotate and blit the image
        surf.blit(rotated_image, origin)

    def eval_outcome(self, win: bool, died: bool):
        """simple function to run the win or die screen after checking won or died"""
        done = False
        if win:
            self.won_screen()
            done = True
            self.reset()

        if died:
            # death_screen()
            self.fill = 0
            self.attempts += 1
            done = True
            self.reset()
        return done

    def won_screen(self):
        """show this screen when beating a level"""
        global attempts, level, fill
        attempts = 0
        self.player_sprite.clear(self.player.image, self.screen)
        self.screen.fill(pygame.Color("yellow"))
        txt_win1 = txt_win2 = "Nothing"
        if self.level == 1:
            if self.coins == 6:
                txt_win1 = f"Coin{self.coins}/6! "
                txt_win2 = "the game, Congratulations"
        else:
            txt_win1 = f"level{self.level}"
            txt_win2 = f"Coins: {self.coins}/6. "
        txt_win = f"{txt_win1} You beat {txt_win2}! Press SPACE to restart, or ESC to exit"

        won_game = self.font.render(txt_win, True, pydash.BLUE)

        self.screen.blit(won_game, (200, 300))
        self.level += 1

        self.wait_for_key()
        self.reset()

    def wait_for_key(self):
        """separate game loop for waiting for a key press while still running game loop"""
        waiting = True
        while waiting:
            self.clock.tick(60)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                        self.reset()
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()

    def move_map(self):
        """moves obstacles along the screen"""
        for sprite in self.elements:
            sprite.rect.x -= self.CameraX

    def draw_stats(self, surf, money=0):

        """
        draws progress bar for level, number of attempts, displays coins collected, and progressively changes progress bar
        colors
        """
        progress_colors = [pygame.Color("red"), pygame.Color("orange"), pygame.Color("yellow"),
                           pygame.Color("lightgreen"),
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

    def check_passed_spike(self):
        #print("The x pos of the player = " + str(self.player_pos))
        #print(self.level_list)
        for row in self.level_list:
            for i in range(1,4):
                if row[self.player_grid_pos-i] == 'Spike':
                    return True
        return False
        #if(self.player.rect.x > self.)
    
    def get_next_spike_distance(self):
        for row in self.level_list:
            for i in range(0,10):
                if row[self.player_grid_pos+i] == 'Spike':
                    return i
        return 10
            
    def Play(self):

        self.player.vel.x = 6
        self.player_pos += self.player.vel.x
        #self.player_pos = [p.rect.x for p in self.player_sprite]

        self.player.update()
        self.CameraX = self.player.vel.x  # for moving obstacles
        self.move_map()  # apply CameraX to all elements
        self.player_grid_pos = round(self.player_pos/32)+6   #35.4 before
        #print(self.player_grid_pos)
        self.passed_spike = self.check_passed_spike()
        self.next_spike_distance = self.get_next_spike_distance()

        if self.render:
            self.alpha_surf.fill((255, 255, 255, 1), special_flags=pygame.BLEND_RGBA_MULT)
            self.screen.blit(self.bg, (0, 0))  # Clear the screen(with the bg)

            # self.player.draw_particle_trail(self.player.rect.left - 1, self.player.rect.bottom + 2,
            #                        pydash.WHITE)
            # self.screen.blit(self.alpha_surf, (0, 0))  # Blit the alpha_surf onto the screen.
            self.draw_stats(self.screen, pydash.coin_count(self.coins))

            if self.player.isjump:
                """rotate the player by an angle and blit it if player is jumping"""
                self.angle -= 8.1712  # this may be the angle needed to do a 360 deg turn in the length covered in one jump by player
                self.blitRotate(self.screen, self.player.image, self.player.rect.center, (16, 16), self.angle)
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
        reward = -0.02
        actions = [action] + [0] * self.step_size
        for action in actions:
            if action == self.JUMP:
                self.player.isjump = True
            elif action == self.NO_JUMP:
                pass
            else:
                raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.Play()
        # Did we win or lose the game?
        done = self.eval_outcome(self.player.win, self.player.died)

        # Null reward everywhere except when reaching the goal (left of the grid)
        if self.player.died == True:
            reward = -5
        elif self.passed_spike:
            reward = 10 

        return self.get_observation(), reward, done

# env = PydashEnv()
# while True:
#     action = env.sample_action()
#     env.step(action)