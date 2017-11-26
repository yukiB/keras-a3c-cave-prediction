# coding:utf-8

import tensorflow as tf
from keras.utils import plot_model
from config import RMSPropDecaly, LEARNING_RATE, MODEL_DIR
from pole_model import create_model
import os


# --グローバルなTensorFlowのDeep Neural Networkのクラスです　-------
class ParameterServer:
    def __init__(self, config):
        self.simple = config.simple
        self.model_name = config.model_name
        
        with tf.variable_scope("parameter_server"):      # スレッド名で重み変数に名前を与え、識別します（Name Space）
            self.model = self._build_model()            # ニューラルネットワークの形を決定

        # serverのパラメータを宣言
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSPropDecaly)    # loss関数を最小化していくoptimizerの定義です
        print(config.simple)

    # 関数名がアンダースコア2つから始まるものは「外部から参照されない関数」、「1つは基本的に参照しない関数」という意味
    def _build_model(self):     # Kerasでネットワークの形を定義します
        model = create_model(self.simple)
        plot_model(model, to_file='A3C.png', show_shapes=True)  # Qネットワークの可視化
        return model

    def save(self, num=0):
        if not os.path.isdir(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        with tf.variable_scope("parameter_server"):
            try:
                json_string = self.model.to_json()
                open(os.path.join(MODEL_DIR,  'model.json'), 'w').write(json_string)
                self.model.save_weights(os.path.join(MODEL_DIR,  'model_' + str(num) + '_weights.hdf5' if num > 0 else 'model_weight.hdf5'))
                print('saved... ' +  'model_' + str(num) + '_weights.hdf5' if num > 0 else 'model_weight.hdf5')
            except:
                pass

    def load(self):
        self.model.load_weights(os.path.join(MODEL_DIR,  self.model_name))
            
