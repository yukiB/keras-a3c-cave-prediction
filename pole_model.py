# coding:utf-8
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from config import NUM_ACTIONS, NUM_WIDTH, NUM_SERIES


def create_model(simple):
    ns = NUM_WIDTH[int(simple)]
    state_input = Input(batch_shape=(None, ns, ns, NUM_SERIES), name='state')
    x = state_input
    x = Conv2D(16, (4, 4), padding='same', activation='relu', strides=(2, 2))(state_input)
    x = Conv2D(32, (2, 2), padding='same', activation='relu', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (2, 2), padding='same', activation='relu', strides=(1, 1))(x)
    x = Conv2D(64, (2, 2), padding='same', activation='relu', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    out_actions = Dense(NUM_ACTIONS, activation='softmax')(x)
    out_value = Dense(1, activation='linear')(x)
    model = Model(inputs=[state_input], outputs=[out_actions, out_value])
    return model
