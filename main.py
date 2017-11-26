# coding:utf-8

from __future__ import absolute_import, with_statement, print_function, unicode_literals
import argparse


argp = argparse.ArgumentParser()
argp.add_argument("-m", "--mode", dest="mode", default="all",
                help="mode (train/test/all (default=all))")
argp.add_argument("--simple", dest="is_simple", action="store_true", default=False,
                help='Train simple model without cnn (8 x 8) (default: off)')
argp.add_argument("-l", "--load", dest="load", default="model_weight.hdf5",
                help='Name of loading weight')

args = argp.parse_args()
mode = str(args.mode)

if not mode in ["train", "test", "all"]:
    print("Please input correct mode.\n")
    print(argp.parse_args('-h'.split()))
    exit()

import tensorflow as tf
import threading
import os
from param_server import ParameterServer
from config import Config, N_WORKERS
from worker_thread  import Worker_thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

conf = Config(args.is_simple, args.load)


with tf.device("/cpu:0"):
    parameter_server = ParameterServer(conf)
    threads = []

    if mode in ["train", "all"]:
        for i in range(N_WORKERS):
            thread_name = "local_thread"+str(i+1)
            threads.append(Worker_thread(thread_name=thread_name, thread_type="learning", parameter_server=parameter_server, config=conf))

    if mode in ["test", "all"]:
        threads.append(Worker_thread(thread_name="test_thread", thread_type="test", parameter_server=parameter_server, config=conf))

    if mode == "test":
        conf.isLearned = True
        parameter_server.load()

COORD = tf.train.Coordinator()
if mode != "test":
    conf.sess.run(tf.global_variables_initializer())    

running_threads = []
for worker in threads:
    job = lambda: worker.run()    
    t = threading.Thread(target=job)
    t.start()
