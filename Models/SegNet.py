import tensorflow.keras as keras
def segnet(
        input_size = (384,384,3)):
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
    #DECONV_BLOCK
    #Block_1
    unpool_1=keras.layers.UpSampling2D(size = (2,2))(pool_5)
    conv_14= keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_1)
    conv_15 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_14)
    conv_16 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_15)
    conv_16= keras.layers.BatchNormalization()(conv_16)
    #Block_2
    unpool_2 = keras.layers.UpSampling2D(size = (2,2))(conv_16)  
    conv_17= keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_2)
    conv_18 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_17)
    conv_19 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_18)
    conv_19= keras.layers.BatchNormalization()(conv_19)
    #Block_3
    unpool_3 =  keras.layers.UpSampling2D(size = (2,2))(conv_19)   
    conv_20= keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_3)
    conv_21 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_20)
    conv_22 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_21)
    conv_22= keras.layers.BatchNormalization()(conv_22)
    #Block_4
    unpool_4 = keras.layers.UpSampling2D(size = (2,2))(conv_22)  
    conv_23= keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_4)
    conv_24 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_23)
    conv_24 = keras.layers.BatchNormalization()(conv_24) 
    #BLock_5
    unpool_5 =keras.layers.UpSampling2D(size = (2,2))(conv_24) 
    conv_25= keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_5)
    conv_26 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_25)
    conv_26 = keras.layers.BatchNormalization()(conv_26)
    
    out=keras.layers.Conv2D(1,1, activation = 'sigmoid', padding = 'same',kernel_initializer = 'he_normal')(conv_26)
    model = keras.models.Model(inputs=inputs, outputs=out, name="SegNet")
    return model