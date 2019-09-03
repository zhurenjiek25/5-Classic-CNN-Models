from keras import layers
from keras.models import Model

def alexnet(in_shape=(227, 227, 3), n_classes=1000, opt='sgd'):
    #227*227*3
    in_layer = layers.Input(in_shape)
    #11*11*96 --> 55*55*96  (227-11)/4 + 1 = 55
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(in_layer)
    #27*27*96               (55-3)/2 +1 = 27
    pool1 = layers.MaxPool2D(pool_size=3, strides=2)(conv1)
    #27*27*256              27/1 = 27
    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)
    #13*13*256              (27-3)/2 +1 = 13
    pool2 = layers.MaxPool2D(pool_size=3, strides=2)(conv2)
    #13*13*384              13/1
    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)
    #13*13*384              13/1
    conv4 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(conv3)
    #13*13*256              13/1
    conv5 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv4)
    #6*6*256                (13-3)/2 +1 = 6
    pool5 = layers.MaxPool2D(pool_size=3, strides=2)(conv5)
    #6*6*256 = 9216
    flat  = layers.Flatten()(pool5)
    #4096
    dense1= layers.Dense(4096, activation='relu')(flat)
    drop1 = layers.Dropout(0.5)(dense1)
    #4096
    dense2= layers.Dense(4096, activation='relu')(drop1)
    drop2 = layers.Dropout(0.5)(dense2)
    #n_classes = 1000
    pred  = layers.Dense(n_classes, activation='softmax')(drop2)

    model = Model(in_layer, pred)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

if __name__=='__main__':
    model = alexnet()
    print(model.summary())
