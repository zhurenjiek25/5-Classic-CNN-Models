#coding:utf-8
from keras import layers
from keras.models import Model

from functools import partial

conv3 = partial(layers.Conv2D,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu')

#执行一种卷积n_convs次
def block(in_tensor, filters, n_convs):
    conv_block = in_tensor
    for _ in range(n_convs):
        conv_block = conv3(filters=filters)(conv_block)
    return conv_block

def _vgg(in_shape=(224, 224, 3), n_classes=1000, opt='sgd', n_stages_per_blocks=[2, 2, 3, 3, 3]):
    in_layer = layers.Input(in_shape)

    block1 = block(in_layer, 64, n_stages_per_blocks[0])
    pool1 = layers.MaxPool2D(pool_size=2, strides=2)(block1)

    block2 = block(pool1, 128, n_stages_per_blocks[1])
    pool2 = layers.MaxPool2D(pool_size=2, strides=2)(block2)

    block3 = block(pool2, 256, n_stages_per_blocks[2])
    pool3 = layers.MaxPool2D(pool_size=2, strides=2)(block3)

    block4 = block(pool3, 512, n_stages_per_blocks[3])
    pool4 = layers.MaxPool2D(pool_size=2, strides=2)(block4)

    block5 = block(pool4, 512, n_stages_per_blocks[4])
    pool5 = layers.MaxPool2D(pool_size=2, strides=2)(block5)

    flat   = layers.Flatten()(pool5)
    dense1 = layers.Dense(4096, activation='relu')(flat)
    dense2 = layers.Dense(4096, activation='relu')(dense1)
    pred   = layers.Dense(1000, activation='softmax')(dense2)

    model = Model(in_layer, pred)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

#2+2+3+3+3 = 13层卷积 + 3层全连接 = 16层
def vgg16(in_shape=(224, 224, 3), n_classes=1000, opt='sgd'):
    return _vgg(in_shape, n_classes, opt)

#2+2+4+4+4 = 16层卷积 + 3层全连接 = 19层
def vgg19(in_shape=(224, 224, 3), n_classes=1000, opt='sgd'):
    return _vgg(in_shape, n_classes, opt, [2, 2, 4, 4, 4])


if __name__=='__main__':
    model = vgg19()
    print(model.summary())