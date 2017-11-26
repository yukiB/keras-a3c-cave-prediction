# coding:utf-8

import time
from env import Environment

# --スレッドになるクラスです　-------
class Worker_thread:
    # スレッドは学習環境environmentを持ちます
    def __init__(self, thread_name, thread_type, parameter_server, config):
        self.environment = Environment(thread_name, thread_type, parameter_server, config)
        self.thread_type = thread_type
        self.parameter_server = parameter_server
        self.config = config

    def run(self):
        while True:
            if not(self.config.isLearned) and self.thread_type is 'learning':     # learning threadが走る
                self.environment.run()

            if not(self.config.isLearned) and self.thread_type is 'test':    # test threadを止めておく
                time.sleep(1.0)

            if self.config.isLearned and self.thread_type is 'learning':     # learning threadを止めておく
                time.sleep(3.0)
                self.parameter_server.save()
                break

            if self.config.isLearned and self.thread_type is 'test':     # test threadが走る
                time.sleep(3.0)
                self.environment.run()
                break


