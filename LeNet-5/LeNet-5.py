#coding:utf-8
from keras import layers
from keras.models import Model

def lenet_5(in_shape=(32, 32, 1), n_classes=10, opt='sgd'):
    #32*32*1
    in_layer = layers.Input(in_shape)
    #5*5*6 --> 28*28*6
    conv1 = layers.Conv2D(filters=6, kernel_size=5, activation='relu')(in_layer)
    #14*14*6
    pool1 = layers.MaxPool2D()(conv1)
    #5*5*16 --> 10*10*16
    conv2 = layers.Conv2D(filters=16, kernel_size=5, activation='relu')(pool1)
    #5*5*16
    pool2 = layers.MaxPool2D()(conv2)
    #5*5*16 = 400
    flatten = layers.Flatten()(pool2)
    #120                                   
    dense1 = layers.Dense(120, activation='relu')(flatten)
    #84
    dense2 = layers.Dense(84, activation='relu')(dense1)
    #n_classes = 10                          500*n_classes + n_classes = 5010
    preds = layers.Dense(n_classes, activation='softmax')(dense2)
    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

if __name__=='__main__':
    model = lenet_5()
    print(model.summary())