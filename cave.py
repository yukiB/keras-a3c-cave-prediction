import os
import numpy as np
import copy
import random
from config import WIN_SIZE


class Block:

    def __init__(self, row, col, size):
        self.row = row
        self.col = col
        self.size = size

    def update(self):
        self.col -= 1

    def hit(self, x, y):
        if self.col + self.size[0] >= x >= self.col:
            if self.row + self.size[1] >= y >= self.row:
                return True
            else:
                return False
        else:
            return False


class Walls:

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.update_size = 0
        self.now_size = 0.55
        self.now_pos = 0.5
        self.now_slope = 0
        self.list = [Wall(n_rows * self.now_size, n_rows * self.now_pos) for c in range(n_cols)]

    def update(self):
        del self.list[0]
        self.update_size += 1
        if self.update_size > 50:
            self.update_size = 0
            self.now_size -= 0.01
            if self.now_size < 0.3:
                self.now_size = 0.3
        val = random.random()  # 0    1     -1
        th_up = 0.6 if self.now_slope != 1 else 0.2  # 0.6 >0.15 >0.6
        th_down = 0.4 if self.now_slope != -1 else 0.8  # 0.4 <0.4 <0.85
        if val > th_up:
            self.now_pos += (val - 0.6) * 0.1
            self.now_slope = 1
        elif val < th_down:
            self.now_pos -= (0.4 - val) * 0.1
            self.now_slope = -1
        else:
            self.now_slope = 0
        self.list.append(Wall(self.n_rows * self.now_size, self.n_rows * self.now_pos))


class Tail:

    def __init__(self, max_size, player):
        self.list = [player]
        self.max_size = max_size
        self.size = player.size
        self.list = [player.y for c in range(max_size)]

    def update(self, player):
        del self.list[0]
        self.list.append(player.y)


class Player:

    def __init__(self, x, y, size):
        self.y_accel = 0.08
        self.y_speed = 0.0
        self.x = x
        self.y = y
        self.size = size

    def update(self, isPushed):
        self.y_accel = -0.08 if isPushed else 0.08
        self.y_speed += self.y_accel
        self.y += self.y_speed


class Wall:

    def __init__(self, size, y_pos):
        self.y = y_pos
        self.size = size


class Cave:

    def __init__(self, time_limit=True):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = WIN_SIZE
        self.screen_n_cols = WIN_SIZE
        self.size = (self.screen_n_rows, self.screen_n_cols)
        self.player_size = 1
        self.player_x = 15
        self.player = Player(self.player_x, self.screen_n_rows / 2, self.player_size)
        self.tail = Tail(self.player_x, self.player)

        self.enable_actions = (0, 1)
        self.frame_rate = 60
        self.state_update_interval = 4
        self.past_time = 0
        self.time_limit = time_limit
        self.block_interval = 20
        self.block_past_time = 0
        self.blocks = []
        self.block_size = [2, 6]

        # variables
        self.reset()

    def update(self, action):
        """
        action:
            0: do nothing
            1: move up
        """
        # update positions
        self.player.update(action)
        self.walls.update()
        self.tail.update(self.player)

        self.reward = 0
        self.terminal = False

        self.past_time += 1
        if self.time_limit and self.past_time > 10000:
            self.terminal = True

        [b.update() for b in self.blocks]

        self.block_past_time += 1
        if self.block_interval < self.block_past_time and self.past_time > 50:
            self.block_past_time = 0
            w = self.walls.list[-1]
            y = np.random.randint(int(w.y - w.size / 2),
                                  int(w.y + w.size / 2 - self.block_size[1]))
            self.blocks.append(Block(y, self.screen_n_cols, self.block_size))

        # check collision
        hits = [b.hit(self.player.x, self.player.y) for b in self.blocks]

        # collision detection
        wall = self.walls.list[self.player.x]
        if (self.player.y <= wall.y - wall.size / 2 or self.player.y >= wall.y + wall.size / 2):
            self.reward = -1
            self.terminal = True
        if sum(hits) > 0:
            self.reward = -1.5
            self.terminal = True

        if len(self.blocks) > 0 and self.blocks[0].col < 0:
            del self.blocks[0]

    def draw(self):
        def _draw_player(player, diff_y):
            self.screen[int(player.y - player.size / 2 - diff_y):int(player.y + player.size / 2 - diff_y),
                        int(player.x - player.size / 2):int(player.x + player.size / 2)] = 1.0

        def _draw_tail(i, max_x, size, y, diff_y):
            self.screen[int(y - size / 2 - diff_y):int(y + size / 2 - diff_y),
                        int(i - size / 2):int(i + size / 2)] = 0.8

        def _draw_block(block, diff_y):
            self.screen[int(block.row - diff_y):int(block.row + block.size[1] - diff_y),
                        block.col:block.col + block.size[0]] = 0.5

        def _draw_wall(i, w, diff_y):
            pos_y = w.y - diff_y
            min_y = int(pos_y - w.size / 2)
            max_y = int(pos_y + w.size / 2)
            min_y = (min_y if min_y > 0 else 0)
            max_y = (max_y if max_y < self.screen_n_rows else self.screen_n_rows)
            self.screen[0:min_y, i] = 0.4
            self.screen[max_y:self.screen_n_rows, i] = 0.4

        # reset screen
        diff_y = self.player.y - self.screen_n_rows / 2.0
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))

        # draw tail
        [_draw_tail(i, self.tail.max_size, self.tail.size, t, diff_y) for i, t in enumerate(self.tail.list)]

        # draw player
        _draw_player(self.player, diff_y)

        # draw block
        [_draw_block(b, diff_y) for b in self.blocks]

        # draw wall
        [_draw_wall(i, w, diff_y) for i, w in enumerate(self.walls.list)]

    def observe(self):
        self.draw()
        return self.screen, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def reset(self):
        # reset cave position
        self.walls = Walls(self.screen_n_rows, self.screen_n_cols)

        # reset player
        self.player = Player(self.player_x, self.screen_n_rows / 2, self.player_size)

        # reset tail
        self.tail = Tail(self.player_x, self.player)

        # reset block
        self.blocks = []
        self.block_past_time = 0

        # reset other variables
        self.reward = 0
        self.terminal = False
        self.past_time = 0
        self.draw()
        return self.screen
