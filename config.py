# coding:utf-8
import tensorflow as tf
import numpy as np


class Config:

    def __init__(self, simple, model_name):
        self.frames = 0
        self.isLearned = False       # 学習が終了したことを示すフラグ
        self.sess = tf.Session()
        self.simple = simple
        self.model_name = model_name


# -- constants of Game
WIN_SIZE = 48
NUM_WIDTH = [WIN_SIZE, WIN_SIZE]  # [conv, simple]
NUM_STATES = [(nw, nw) for nw in NUM_WIDTH]
NUM_ACTIONS = 2
NONE_STATE = [np.zeros(ns) for ns in NUM_STATES]

# -- params of Advantage-ベルマン方程式
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

# -- constants of LocalBrain
MIN_BATCH = 5
LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
LEARNING_RATE =  0.00015
RMSPropDecaly = 0.99

Tmax = 10   # 各スレッドの更新ステップ間隔

N_WORKERS = 8   # スレッドの数

# ε-greedyのパラメータ
EPS_START = 1.0
EPS_END = 0.1
EPS_STEPS = [15000 * N_WORKERS, 10000 * N_WORKERS]

MODEL_DIR = "model"

NUM_SERIES = 4
