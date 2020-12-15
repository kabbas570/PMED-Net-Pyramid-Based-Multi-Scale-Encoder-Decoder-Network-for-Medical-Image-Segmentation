import tensorflow.keras as keras
def fcn_8(input_size = (384,384,3)): 
     # Block 1
    inputs = keras.layers.Input(input_size)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1',kernel_initializer = 'he_normal')(inputs)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv1',kernel_initializer = 'he_normal')(pool_1)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
     # Block 3
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1',kernel_initializer = 'he_normal')(pool_2)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block3_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1',kernel_initializer = 'he_normal')(pool_3)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1',kernel_initializer = 'he_normal')(pool_4)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu', padding='same',name='block5_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_5 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    # up convolutions and sum
    pool_5_U=keras.layers.UpSampling2D(size = (4,4),interpolation='bilinear')(pool_5)
    pool_5_U=keras.layers.BatchNormalization()(pool_5_U)
    pool_4_U=keras.layers.UpSampling2D(size = (2,2),interpolation='bilinear')(pool_4)
    pool_4_U=keras.layers.BatchNormalization()(pool_4_U)
    sum_1=keras.layers.Add()([pool_5_U,pool_4_U])  
    sum_2=keras.layers.Add()([sum_1,pool_3])  
    sum_2=keras.layers.BatchNormalization()(sum_2)
    x=keras.layers.UpSampling2D(size = (8,8),interpolation='bilinear')(sum_2)  
    x =keras.layers.Conv2D(2, 1, activation = 'relu',kernel_initializer = 'he_normal')(x)
    x =keras.layers.Conv2D(1, 1, activation = 'sigmoid',kernel_initializer = 'he_normal')(x)
    model = keras.models.Model(inputs = inputs, outputs = x)
    return model
