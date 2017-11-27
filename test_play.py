from __future__ import division

import argparse
import os
import numpy as np
import sys
from multiprocessing import Pool
import multiprocessing as multi

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time
from cave import Cave


def init():
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


def gray2rgb(val):
    if val == 0.4:
        return np.array([0.0, 0.8, 1.0])
    if val == 0.5:
        return np.array([0.0, 0.4, 1.0])
    elif val >= 0.8:
        return np.array([1.0, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, 0.0])


def animate(step):
    global past_time
    global state_t_1, reward_t, terminal, action_t, past_time

    if terminal:
        env.reset()
        sys.stderr.write('\r\033[K')
        print("SCORE: {0:03d}".format(past_time))
        past_time = 0
    else:
        state_t = state_t_1
        past_time += 1
        sys.stderr.write('\r\033[K SCORE: {0:03d}'.format(past_time))

        # execute action in environment
        # action_t = agent.select_action(state_t, 0.0)
        env.execute_action(action_t)

    # observe environment
    state_t_1, reward_t, terminal = env.observe()
    # animate
    #img.set_array(state_t_1)
    shape = state_t_1.shape
    data = state_t_1.ravel()
    p = Pool(multi.cpu_count())
    data = p.map(gray2rgb, data)
    p.close()
    img.set_array(np.array(data).reshape(*shape, 3))
    plt.axis("off")
    return img,


def onkey(event):
    global action_t
    if event.key == ' ':
        action_t = 1
    elif event.key == 'q':
        time.sleep(1)
        plt.close('all')
        sys.exit()


def offkey(event):
    global action_t
    if event.key == ' ':
        action_t = 0


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()
    print('----------------------')
    print('space: moving up')
    print('q:     exit')
    print('----------------------')

    # environmet, agent
    env = Cave(time_limit=False)

    # variables
    past_time = 0
    state_t_1, reward_t, terminal = env.observe()
    action_t = 0
    
    # animate
    fig = plt.figure(figsize=(env.screen_n_rows / 10, env.screen_n_cols / 10))
    fig.canvas.set_window_title("{}".format(env.name))
    cid = fig.canvas.mpl_connect('key_press_event', onkey)
    cid = fig.canvas.mpl_connect('key_release_event', offkey)
    img = plt.imshow(state_t_1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(1000 / env.frame_rate), blit=True)

    plt.show()
