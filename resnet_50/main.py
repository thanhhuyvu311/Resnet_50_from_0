import tensorflow as tf


def Conv_block(x,filter,stride):
    """

    :param x: input
    :param filter: so luong filter
    :param stride: buoc nhay
    :return:

    day la block se lam thay doi kich thuoc input, tac dung na na voi max pool

    conv_block se thuc hien quy trinh sau:
        1. conv kernel 1x1 se giam kich thuoc filter
        2. thuc hien tich chap voi so filter tai buoc 1
        3. conv kernel 1x1 de lam tang kich thuoc filter
    """

    x_skip = x
    f1,f2 = filter #ky thuat tach tuples (x,y) => f1=x, f2 = y

    #block dau tien
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1,1), strides=(stride,stride), padding='valid', kernel_regularizer = tf.keras.regularizers.l2(0.001))(x)
    #neu stride = 2 thi no se la viec lam giam kich thuoc cua ban do dac trung
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #block thu 2
    x = tf.keras.layers.Conv2D(filters=f1,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer = tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #block thu 3 (tang so kenh ban do dac trung)
    x = tf.keras.layers.Conv2D(filters=f2,kernel_size=(1,1),strides=(1,1),padding='valid',kernel_regularizer = tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #tang so kenh x_skip = voi x tai block 3
    x_skip = tf.keras.layers.Conv2D(filters=f2,kernel_size=(1,1),strides=(1,1),padding='valid',kernel_regularizer = tf.keras.regularizers.l2(0.001))(x)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    #cong x_skip vao x
    x = tf.keras.layers.Add()([x,x_skip]) #thuc hien cong 2 dau ra
    x = tf.keras.layers.ReLU()(x)

    return x


def Res_id_block(x,filter):
    """

    :param x: input
    :param filter: so luong kenh
    :return:

    resnet block se khong thay doi kich thuoc tai block nay
    res_id_block se thuc hien qua trinh sau
    1. thuc hien tich chap 1x1 voi stride = 1
    2. thuc hien tich chap 3x3
    3. thuc hien tich chap 1x1 voi stride = 1
    """

    x_skip = x
    f1,f2 = filter

    #block 1
    x = tf.keras.layers.Conv2D(filters=f1,kernel_size=(1,1),padding='valid',kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #block 2
    x = tf.keras.layers.Conv2D(filters=f1,kernel_size=(3,3),padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #block 3
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), padding='valid',kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    #add
    x = tf.keras.layers.Add()([x,x_skip])
    x = tf.keras.layers.ReLU()(x)
    return x

def resnet_50():
    #input size 224x224x3
    input_dim = tf.keras.layers.Input(shape=(224,224,3))

    #su dung padding de giu lai bien anh truoc khi vao conv2d 7x7
    x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(input_dim)

    #khoi 1
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(7,7),padding='valid',use_bias=False,name='Conv7x7')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),name='overlapping')(x)

    #khoi 2
    x = Conv_block(x,(64,256),2) #giam kich thuoc anh xuong 2 lan
    x = Res_id_block(x,(64,256))
    x = Res_id_block(x,(64,256))

    #khoi 3
    x = Conv_block(x,(128,512),2)
    x = Res_id_block(x,(128,512))
    x = Res_id_block(x,(128,512))
    x = Res_id_block(x,(128,512))

    #khoi 4
    x = Conv_block(x,(256,1024),2)
    x = Res_id_block(x,(256,1024))
    x = Res_id_block(x,(256,1024))
    x = Res_id_block(x,(256, 1024))
    x = Res_id_block(x,(256, 1024))
    x = Res_id_block(x,(256, 1024))
    x = Res_id_block(x,(256, 1024))

    #khoi 5
    x = Conv_block(x,(512,2048),2)
    x = Res_id_block(x,(512,2048))
    x = Res_id_block(x, (512, 2048))

    x = tf.keras.layers.AveragePooling2D((2,2),padding='same')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(100,activation='softmax',kernel_initializer='he_normal')(x)

    model = tf.keras.models.Model(inputs = input_dim,outputs = x, name = 'resnet_50')

    return model
