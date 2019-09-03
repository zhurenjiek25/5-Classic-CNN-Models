#coding:utf-8
from keras import layers
from keras.models import Model

#先batch normalization 再 activation
def _after_conv(in_tensor, active=True):
    norm = layers.BatchNormalization()(in_tensor)
    if active:
        return layers.Activation('relu')(norm)
    else:
        return norm

#1x1卷积后batch normalization 然后activation
def conv1(in_tensor, filters, active=True):
    conv = layers.Conv2D(filters, kernel_size=1, strides=1)(in_tensor)
    return _after_conv(conv, active)

#步长为2 的1x1卷积
def conv1_downsample(in_tensor, filters, active=True):
    conv = layers.Conv2D(filters, kernel_size=1, strides=2)(in_tensor)
    return _after_conv(conv, active)

#3x3卷积后batch normalization 然后activation
def conv3(in_tensor, filters, active=True):
    conv = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(in_tensor)
    return _after_conv(conv, active)

#步长为2 的3x3卷积
def conv3_downsample(in_tensor, filters, active=True):
    conv = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(in_tensor)
    return _after_conv(conv, active)

#type one bottleneck, 3x3 conv twice
def resnet_block_wo_bottleneck(in_tensor, filters, downsample=False):
    if downsample:
        conv1_rb = conv3_downsample(in_tensor, filters)
    else:
        conv1_rb = conv3(in_tensor, filters)
    conv2_rb = conv3(conv1_rb, filters, active=False)#no activation

    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters, active=False)#no activation

    result = layers.Add()([conv2_rb, in_tensor])
    
    return layers.Activation('relu')(result)# activation after add

#type two bottleneck, 1x1, 3x3, 1x1
def resnet_block_w_bottleneck(in_tensor, 
                             filters, 
                             downsample=False,
                             change_channels=False):
    if downsample:
        conv1_rb = conv1_downsample(in_tensor, int(filters/4))
    else:
        conv1_rb = conv1(in_tensor, int(filters/4))
    
    conv2_rb = conv3(conv1_rb, int(filters/4))
    conv3_rb = conv1(conv2_rb, filters, active=False)#no activation

    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters, active=False)
    elif change_channels:
        in_tensor = conv1(in_tensor, filters, active=False)
    
    result = layers.Add()([conv3_rb, in_tensor])

    return layers.Activation('relu')(result)# activation after add

#first convolution block
def _pre_res_block(in_tensor):
    conv = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(in_tensor)
    conv = _after_conv(conv)
    pool = layers.MaxPool2D(3, 2, padding='same')(conv)
    return pool

#last FC block
def _post_res_block(in_tensor, n_classes):
    pool = layers.GlobalAvgPool2D()(in_tensor)
    preds = layers.Dense(n_classes, activation='softmax')(pool)
    return preds

#type one bottleneck convolution serveral times
def convx_wo_bottleneck(in_tensor, filters, n_times, downsample_1=False):
    res = in_tensor
    for i in range(n_times):
        if i == 0:
            res = resnet_block_wo_bottleneck(res, filters, downsample_1)
        else:
            res = resnet_block_wo_bottleneck(res, filters)
    return res

#type two bottleneck convolution
def convx_w_bottleneck(in_tensor, filters, n_times, downsample_1=False):
    res = in_tensor
    for i in range(n_times):
        if i == 0:
            res = resnet_block_w_bottleneck(res, filters, downsample_1, not downsample_1)
        else:
            res = resnet_block_w_bottleneck(res, filters)
    return res


def _resnet(in_shape=(224,224,3),
            n_classes=1000,
            opt='sgd',
            convx=[64,128,256,512],
            n_convx=[2,2,2,2],
            convx_fn=convx_wo_bottleneck):
    in_layer = layers.Input(in_shape)
    
    downsampled = _pre_res_block(in_layer)
    
    conv2x = convx_fn(downsampled, convx[0], n_convx[0], True)
    conv3x = convx_fn(downsampled, convx[1], n_convx[1], True)
    conv4x = convx_fn(downsampled, convx[2], n_convx[2], True)
    conv5x = convx_fn(downsampled, convx[3], n_convx[3], True)

    preds = _post_res_block(conv5x, n_classes)

    model = Model(in_layer, preds)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def resnet18(in_shape=(224, 224, 3),
             n_classes=1000,
             opt='sgd'):
    return _resnet(in_shape, n_classes, opt)

def resnet34(in_shape=(224, 224, 3),
             n_classes=1000,
             opt='sgd'):
    return _resnet(in_shape, n_classes, opt, n_convx=[3, 4, 6, 3])

def resnet50(in_shape=(224, 224, 3),
             n_classes=1000,
             opt='sgd'):
    return _resnet(in_shape, n_classes, opt, convx=[265,512,1024,2048], n_convx=[3,4,6,3],convx_fn=convx_w_bottleneck)

def resnet101(in_shape=(224, 224, 3),
             n_classes=1000,
             opt='sgd'):
    return _resnet(in_shape, n_classes, opt, convx=[265,512,1024,2048], n_convx=[3,4,23,3],convx_fn=convx_w_bottleneck)

def resnet152(in_shape=(224, 224, 3),
             n_classes=1000,
             opt='sgd'):
    return _resnet(in_shape, n_classes, opt, convx=[265,512,1024,2048], n_convx=[3,8,36,3],convx_fn=convx_w_bottleneck)

if __name__=='__main__':
    model = resnet50()
    print(model.summary())