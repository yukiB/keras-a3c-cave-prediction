# coding:utf-8

import time
import numpy as np
from agent import Agent
from config import Tmax
from cave import Cave
import matplotlib.animation as animation
from multiprocessing import Pool
import multiprocessing as multi
import matplotlib.pyplot as plt
from collections import deque
from config import NUM_SERIES

state_num = NUM_SERIES

# --CartPoleを実行する環境です、TensorFlowのスレッドになります　-------

def gray2rgb(val):
    if val == 0.4:
        return np.array([0.0, 0.8, 1.0])
    if val == 0.5:
        return np.array([0.0, 0.4, 1.0])
    elif val >= 0.8:
        return np.array([1.0, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, 0.0])    

def onkey(event):
    if event.key == 'q':
        time.sleep(1)
        plt.close('all')
        sys.exit()

class Environment:
    total_reward_vec = np.zeros(10)  # 総報酬を10試行分格納して、平均総報酬をもとめる
    count_trial_each_thread = 0     # 各環境の試行数

    def __init__(self, name, thread_type, parameter_server, config):
        self.name = name
        self.thread_type = thread_type
        self.env = Cave()
        self.agent = Agent(name, parameter_server, config, thread_type)    # 環境内で行動するagentを生成
        self.config = config
        self.parameter_server = parameter_server

    def run(self):
        self.agent.brain.pull_parameter_server()

        if self.thread_type is 'test':
            self.state_t_1, self.reward_t, self.terminal = self.env.observe()
            fig = plt.figure(figsize=(self.env.screen_n_rows / 10, self.env.screen_n_cols / 10))
            fig.canvas.set_window_title("{}".format(self.env.name))
            cid = fig.canvas.mpl_connect('key_press_event', onkey)
            self.img = plt.imshow(self.state_t_1, interpolation="none", cmap="gray")
            self.ani = animation.FuncAnimation(
                fig, self.animate, init_func=self.init, interval=(1000 / self.env.frame_rate), blit=True)
            plt.show()
            return None

        s = self.env.reset()
        s = [s] * state_num
        s = np.array(s).transpose((2, 1, 0)).transpose((1, 0, 2))
        R = 0
        step = 0
        S = deque(maxlen=state_num * 2)
        while True:

            a = self.agent.act(s)   # 行動を決定
            self.env.execute_action(a)
            s_, r, done = self.env.observe()

            step += 1
            self.config.frames += 1     # セッショントータルの行動回数をひとつ増やします

            if len(S) == 0:
                [S.append(s_) for i in range(state_num * 2)]
            else:
                S.append(s_)

            tmpS = [S[(ss + 1) * 2 - 1] for ss in range(state_num)]
            tmpS = np.array(tmpS).transpose((2, 1, 0)).transpose((1, 0, 2))
            
            if not(self.config.isLearned) and self.thread_type is 'learning':
                self.agent.advantage_push_local_brain(s, a, r, tmpS)
            s = tmpS
            R += r
            if done or (step % Tmax == 0):  # 終了時がTmaxごとに、parameterServerの重みを更新し、それをコピーする
                if not(self.config.isLearned) and self.thread_type is 'learning':
                    self.agent.brain.update_parameter_server()
                    self.agent.brain.pull_parameter_server()

            if done:
                self.total_reward_vec = np.hstack(
                    (self.total_reward_vec[1:], step))  # トータル報酬の古いのを破棄して最新10個を保持
                self.count_trial_each_thread += 1  # このスレッドの総試行回数を増やす
                break
        # 総試行数、スレッド名、今回の報酬を出力
        print("スレッド：" + self.name + "、試行数：" + str(self.count_trial_each_thread) +
              "、今回のステップ:" + str(step) + "、平均ステップ：" + str(self.total_reward_vec.mean()) + ", トータルステップ：" + str(self.config.frames))

        # スレッドで平均報酬が一定を越えたら終了
        if self.name == "local_thread1" and self.count_trial_each_thread % 500 == 0:
            self.parameter_server.save(self.count_trial_each_thread)
        if self.total_reward_vec.mean() > 799:
            self.config.isLearned = True
            time.sleep(2.0)     # この間に他のlearningスレッドが止まります
            self.agent.brain.push_parameter_server()    # この成功したスレッドのパラメータをparameter-serverに渡します

    def init(self):
        self.img.set_array(self.state_t_1)
        self.n_count = 0
        plt.axis("off")
        self.S = deque(maxlen=state_num * 2)
        return self.img,

    def animate(self, step):

        if self.terminal:
            self.env.reset()
            self.S = None
            self.S = deque(maxlen=state_num * 2)
            print("SCORE: {0:03d}".format(self.n_count))
            self.n_count = 0
        else:
            state_t = self.state_t_1
            self.n_count += 1
            if self.reward_t == 1:
                self.n_catched += 1
                # execute action in environment

            if len(self.S) == 0:
                [self.S.append(state_t) for i in range(state_num * 2)]
            else:
                self.S.append(state_t)

            tmpS = [self.S[(ss + 1) * 2 - 1] for ss in range(state_num)]
            tmpS = np.array(tmpS).transpose((2, 1, 0)).transpose((1, 0, 2))                
            action_t = self.agent.act_test(tmpS)
            self.env.execute_action(action_t)

        # observe environment
        self.state_t_1, self.reward_t, self.terminal = self.env.observe()

        # animate
        shape = self.state_t_1.shape
        data = self.state_t_1.ravel()
        p = Pool(multi.cpu_count())
        data = p.map(gray2rgb, data)
        p.close()
        self.img.set_array(np.array(data).reshape(*shape, 3))    
        plt.axis("off")
        return self.img,
