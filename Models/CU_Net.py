import tensorflow.keras as keras
def CU_net():
    inputs = keras.layers.Input((384,384,3))
    conv1 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1=keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1=keras.layers.BatchNormalization()(conv1)
    conv1x1 = keras.layers.Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    m1 = keras.layers.Add()([conv1x1,conv1])
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(m1)
    conv2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2=keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2=keras.layers.BatchNormalization()(conv2)
    conv1x1 = keras.layers.Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    
    m2 = keras.layers.Add()([conv1x1,conv2])
    
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(m2)
    
    
    conv3 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3=keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=keras.layers.BatchNormalization()(conv3)
    conv1x1 = keras.layers.Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    
    m3 = keras.layers.Add()([conv1x1,conv3])
    
    
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(m3)
    conv4 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4=keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=keras.layers.BatchNormalization()(conv4)
    conv1x1 = keras.layers.Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    m4 = keras.layers.Add()([conv1x1,conv4])
    
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(m4)
    
    
    conv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5_=keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5_=keras.layers.BatchNormalization()(conv5)
    conv1x1 = keras.layers.Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    m5 = keras.layers.Add()([conv1x1,conv5])
    
    up6 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(m5))
    merge6 = keras.layers.concatenate([conv4,up6], axis = 3)
    conv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6=  keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6=  keras.layers.BatchNormalization()(conv6)
    conv1x1 = keras.layers.Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    m6 = keras.layers.Add()([conv1x1,conv6])
    
    up7 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(m6))
    merge7 = keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7=keras.layers.BatchNormalization()(conv7)
    conv7 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7=keras.layers.BatchNormalization()(conv7)
    conv1x1 = keras.layers.Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    m7 = keras.layers.Add()([conv1x1,conv7])
    
    up8 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(m7))
    merge8 = keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8=keras.layers.BatchNormalization()(conv8)
    conv8 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8=keras.layers.BatchNormalization()(conv8)
    conv1x1 = keras.layers.Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    m8= keras.layers.Add()([conv1x1,conv8])
    
    up9 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(m8))
    merge9 = keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9=keras.layers.BatchNormalization()(conv9)
    conv9 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9=keras.layers.BatchNormalization()(conv9)
    
    conv1x1 = keras.layers.Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    m9= keras.layers.Add()([conv1x1,conv9])
    
    out1=keras.layers.Conv2D(1, 1, activation = 'sigmoid')(m9)
    
    conv10 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(m9)
    conv10=keras.layers.BatchNormalization()(conv10)
    conv10= keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    conv10=keras.layers.BatchNormalization()(conv10)
    
    conv1x1 = keras.layers.Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(m9)
    
    m2 = keras.layers.Add()([conv1x1,conv10])
    
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(m2)
    
    merge = keras.layers.concatenate([conv8,pool2], axis = 3)
    
    conv3 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
    conv3=keras.layers.BatchNormalization()(conv3)
    conv3= keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=keras.layers.BatchNormalization()(conv3)
    conv1x1 = keras.layers.Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
    
    m2 = keras.layers.Add()([conv1x1,conv3])
    
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(m2)
    
    merge = keras.layers.concatenate([conv7,pool3], axis = 3)

    conv4 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
    conv4=keras.layers.BatchNormalization()(conv4)
    conv4= keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=keras.layers.BatchNormalization()(conv4)
    
    conv1x1 = keras.layers.Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
    
    m2 = keras.layers.Add()([conv1x1,conv4])
    
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    merge = keras.layers.concatenate([conv6,pool4], axis = 3)
    
    conv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5=keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5=keras.layers.BatchNormalization()(conv5)
    
    conv1x1 = keras.layers.Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    m2 = keras.layers.Add()([conv1x1,conv5])
    
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(m2)
    conc=keras.layers.concatenate([pool5,conv5_], axis = 3)
    up6 = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conc))
    merge6 = keras.layers.concatenate([pool4,up6], axis = 3)
    conv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6=  keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6=  keras.layers.BatchNormalization()(conv6)
    conv1x1 = keras.layers.Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    m6 = keras.layers.Add()([conv1x1,conv6])
    up7 = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(m6))
    merge7 = keras.layers.concatenate([pool3,up7], axis = 3)
    conv7 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7=keras.layers.BatchNormalization()(conv7)
    conv7 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7=keras.layers.BatchNormalization()(conv7)
    conv1x1 = keras.layers.Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    m7 = keras.layers.Add()([conv1x1,conv7])
    up8 = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(m7))
    merge8 = keras.layers.concatenate([pool2,up8], axis = 3)
    conv8 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8=keras.layers.BatchNormalization()(conv8)
    conv8 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8=keras.layers.BatchNormalization()(conv8)
    conv1x1 = keras.layers.Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    m8= keras.layers.Add()([conv1x1,conv8])
    up9 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(m8))
    merge9 = keras.layers.concatenate([conv9,up9], axis = 3)
    conv9 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9=keras.layers.BatchNormalization()(conv9)
    conv9 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9=keras.layers.BatchNormalization()(conv9)
    conv1x1 = keras.layers.Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    m9= keras.layers.Add()([conv1x1,conv9]) 
    out2=keras.layers.Conv2D(1, 1, activation = 'sigmoid')(m9)
    model = keras.models.Model(inputs = inputs, outputs = [out1,out2])
    return model