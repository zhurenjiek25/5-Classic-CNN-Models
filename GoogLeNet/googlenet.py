from keras import layers
from keras.models import Model
from functools import partial

conv1x1 = partial(layers.Conv2D, kernel_size=1, activation='relu')
conv3x3 = partial(layers.Conv2D, kernel_size=3, padding='same', activation='relu')
conv5x5 = partial(layers.Conv2D, kernel_size=5, padding='same', activation='relu')

#inception 模块
def inception_module(in_tensor, n1, n3_1, n3, n5_1, n5, pp):
    conv1 = conv1x1(n1)(in_tensor)
    
    conv3_1 = conv1x1(n3_1)(in_tensor)
    conv3 = conv3x3(n3)(conv3_1)

    conv5_1 = conv1x1(n5_1)(in_tensor)
    conv5 = conv5x5(n5)(conv5_1)

    #从论文上看，应该是先maxpool 再1x1卷积
    pool_conv = conv1x1(pp)(in_tensor)
    pool = layers.MaxPool2D(3, strides=1, padding='same')(pool_conv)

    merged = layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool])
    return merged

#辅助分类模块
def aux_clf(in_tensor, n_classes=1000):
    avg_pool = layers.AvgPool2D(5, 3)(in_tensor)
    conv = conv1x1(128)(avg_pool)
    flat = layers.Flatten()(conv)
    dense = layers.Dense(1024, activation='relu')(flat)
    drop = layers.Dropout(0.7)(dense)
    out = layers.Dense(n_classes, activation='softmax')(drop)
    return out

def inception_v1(in_shape=(224, 224, 3), n_classes=1000, opt='sgd'):
    in_layer = layers.Input(in_shape)
    
    conv1 = layers.Conv2D(64, 7, strides=2, activation='relu', padding='same')(in_layer)
    #pool1 = layers.MaxPool2D(3, 2, padding='same')(conv1)
    pad1 = layers.ZeroPadding2D()(conv1)
    pool1 = layers.MaxPool2D(3, 2)(pad1)
    
    conv2 = layers.Conv2D(192, 3, strides=1, activation='relu', padding='same')(pool1)
    pad2 = layers.ZeroPadding2D()(conv2)
    pool2 = layers.MaxPool2D(3, 2)(pad2)

    inception_3a = inception_module(pool2, 64, 96, 128, 16, 32, 32)
    inception_3b = inception_module(inception_3a, 128, 128, 192, 32, 96, 64)
    pad3 = layers.ZeroPadding2D()(inception_3b)
    pool3 = layers.MaxPool2D(3, 2)(pad3)

    inception_4a = inception_module(pool3, 192, 96, 208, 16, 48, 64)
    inception_4b = inception_module(inception_4a, 160, 112, 224, 24, 64, 64)
    inception_4c = inception_module(inception_4b, 128, 128, 256, 24, 64, 64)
    inception_4d = inception_module(inception_4c, 112, 144, 288, 32, 64, 64)
    inception_4e = inception_module(inception_4d, 256, 160, 320, 32, 128, 128)
    pad4 = layers.ZeroPadding2D()(inception_4e)
    pool4 = layers.MaxPool2D(3, 2)(pad4)

    aux0 = aux_clf(inception_4a, n_classes)
    aux1 = aux_clf(inception_4d, n_classes)

    inception_5a = inception_module(pool4, 256, 160, 320, 32, 128, 128)
    inception_5b = inception_module(inception_5a, 384, 192, 384, 48, 128, 128)

    #pad5 = layers.ZeroPadding2D()(inception_5b)
    pool5 = layers.AvgPool2D(7, 1)(inception_5b)
    drop = layers.Dropout(0.4)(pool5)
    flat = layers.Flatten()(drop)

    preds = layers.Dense(n_classes, activation='softmax')(flat)

    model = Model(in_layer, [preds, aux0, aux1])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

if __name__=='__main__':
    model = inception_v1()
    print(model.summary())


