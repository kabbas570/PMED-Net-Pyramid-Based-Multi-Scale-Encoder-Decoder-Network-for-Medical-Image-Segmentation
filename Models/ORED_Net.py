import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #print(updates.shape)
            #print(mask.shape)
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3])

            ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                                  K.flatten(updates),
                                  [K.prod(output_shape)])

            input_shape = updates.shape
            out_shape = [-1,
                         input_shape[1] * self.size[0],
                         input_shape[2] * self.size[1],
                         input_shape[3]]
        return K.reshape(ret, out_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
                )
         
def ORED_net(input_size = (384,384,3)):
    inputs = keras.layers.Input(input_size)
    E_Conv_1_1 = keras.layers.Conv2D(64, 3,  padding = 'same', kernel_initializer = 'he_normal')(inputs)
    E_Conv_1_1=keras.layers.BatchNormalization()(E_Conv_1_1)  
    E_Conv_1_1=keras.layers.Activation('relu')(E_Conv_1_1)
    ORED_P_1=keras.layers.Conv2D(64, 1,  padding = 'same', kernel_initializer = 'he_normal')(E_Conv_1_1) 
    ORED_P_1=keras.layers.BatchNormalization()(ORED_P_1)  
    E_Conv_1_2 = keras.layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(E_Conv_1_1)
    
    E_Conv_1_2=keras.layers.BatchNormalization()(E_Conv_1_2)
    E_Conv_1_2=keras.layers.Activation('relu')(E_Conv_1_2)
    #pool_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(E_Conv_1_2) 
    pool_1,indices_1=tf.nn.max_pool_with_argmax( E_Conv_1_2, 2, 2,'SAME', data_format='NHWC', include_batch_in_index=True, name=None)
    
    E_Conv_2_1 = keras.layers.Conv2D(128, 3,  padding = 'same', kernel_initializer = 'he_normal')(pool_1)
    E_Conv_2_1=keras.layers.BatchNormalization()(E_Conv_2_1)
    E_Conv_2_1=keras.layers.Activation('relu')(E_Conv_2_1)
    ORED_P_2=keras.layers.Conv2D(128, 1,  padding = 'same', kernel_initializer = 'he_normal')(E_Conv_2_1)  
    ORED_P_2=keras.layers.BatchNormalization()(ORED_P_2)
    E_Conv_2_2 = keras.layers.Conv2D(128, 3,  padding = 'same', kernel_initializer = 'he_normal')(E_Conv_2_1)
    E_Conv_2_2=keras.layers.BatchNormalization()(E_Conv_2_2)
    E_Conv_2_2=keras.layers.Activation('relu')(E_Conv_2_2)
    pool_2,indices_2=tf.nn.max_pool_with_argmax( E_Conv_2_2, 2, 2,'SAME', data_format='NHWC', include_batch_in_index=True, name=None)


    E_Conv_3_1 = keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool_2)
    E_Conv_3_1=keras.layers.BatchNormalization()(E_Conv_3_1)   
    E_Conv_3_1=keras.layers.Activation('relu')(E_Conv_3_1)
    ORED_P_3=keras.layers.Conv2D(256, 1,  padding = 'same', kernel_initializer = 'he_normal')(E_Conv_3_1) 
    ORED_P_3=keras.layers.BatchNormalization()(ORED_P_3)
    E_Conv_3_2 = keras.layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(E_Conv_3_1)
    E_Conv_3_2=keras.layers.BatchNormalization()(E_Conv_3_2)
    E_Conv_3_2=keras.layers.Activation('relu')(E_Conv_3_2) 
    pool_3,indices_3=tf.nn.max_pool_with_argmax( E_Conv_3_2, 2, 2,'SAME', data_format='NHWC', include_batch_in_index=True, name=None)

    
    E_Conv_4_1 = keras.layers.Conv2D(512, 3,padding = 'same', kernel_initializer = 'he_normal')(pool_3)
    E_Conv_4_1=keras.layers.BatchNormalization()(E_Conv_4_1)   
    E_Conv_4_1=keras.layers.Activation('relu')(E_Conv_4_1)
    ORED_P_4=keras.layers.Conv2D(512, 1, padding = 'same', kernel_initializer = 'he_normal')(E_Conv_4_1) 
    ORED_P_4=keras.layers.BatchNormalization()(ORED_P_4)
    E_Conv_4_2 = keras.layers.Conv2D(512, 3,  padding = 'same', kernel_initializer = 'he_normal')(E_Conv_4_1)
    E_Conv_4_2=keras.layers.BatchNormalization()(E_Conv_4_2)
    E_Conv_4_2=keras.layers.Activation('relu')(E_Conv_4_2)
    pool_4,indices_4=tf.nn.max_pool_with_argmax( E_Conv_4_2, 2, 2,'SAME', data_format='NHWC', include_batch_in_index=True, name=None)
        
    max_unpool_4 = MaxUnpooling2D()([pool_4,indices_4])
    ##Decoder Part
    D_Conv_4_1 = keras.layers.Conv2D(512, 3,padding = 'same', kernel_initializer = 'he_normal')(max_unpool_4)
    D_Conv_4_1=keras.layers.BatchNormalization()(D_Conv_4_1)   
    D_Conv_4_1=keras.layers.Activation('relu')(D_Conv_4_1)
    
    Add_4 = keras.layers.add([D_Conv_4_1, ORED_P_4])
    D_Conv_4_2 = keras.layers.Conv2D(256, 3,padding = 'same', kernel_initializer = 'he_normal')(Add_4)
    D_Conv_4_2=keras.layers.BatchNormalization()(D_Conv_4_2)   
    D_Conv_4_2=keras.layers.Activation('relu')(D_Conv_4_2)
    
    max_unpool_3 = MaxUnpooling2D()([D_Conv_4_2,indices_3])
    D_Conv_3_1 = keras.layers.Conv2D(256, 3,padding = 'same', kernel_initializer = 'he_normal')(max_unpool_3)
    D_Conv_3_1=keras.layers.BatchNormalization()(D_Conv_3_1)   
    D_Conv_3_1=keras.layers.Activation('relu')(D_Conv_3_1)
    Add_3 = keras.layers.add([D_Conv_3_1, ORED_P_3])
    
    D_Conv_3_2 = keras.layers.Conv2D(128, 3,padding = 'same', kernel_initializer = 'he_normal')(Add_3)
    D_Conv_3_2=keras.layers.BatchNormalization()(D_Conv_3_2)   
    D_Conv_3_2=keras.layers.Activation('relu')(D_Conv_3_2)
    #print(indices_2)
    max_unpool_2 = MaxUnpooling2D()([D_Conv_3_2,indices_2])
    D_Conv_2_1 = keras.layers.Conv2D(128, 3,padding = 'same', kernel_initializer = 'he_normal')(max_unpool_2)
    D_Conv_2_1=keras.layers.BatchNormalization()(D_Conv_2_1)   
    D_Conv_2_1=keras.layers.Activation('relu')(D_Conv_2_1)
    Add_2 = keras.layers.add([D_Conv_2_1, ORED_P_2])
    
    D_Conv_2_2 = keras.layers.Conv2D(64, 3,padding = 'same', kernel_initializer = 'he_normal')(Add_2)
    D_Conv_2_2=keras.layers.BatchNormalization()(D_Conv_2_2)   
    D_Conv_2_2=keras.layers.Activation('relu')(D_Conv_2_2)
    #print(D_Conv_2_2)
    max_unpool_1 = MaxUnpooling2D()([D_Conv_2_2,indices_1])
    D_Conv_1_1 = keras.layers.Conv2D(64, 3,padding = 'same', kernel_initializer = 'he_normal')(max_unpool_1)
    D_Conv_1_1=keras.layers.BatchNormalization()(D_Conv_1_1)   
    D_Conv_1_1=keras.layers.Activation('relu')(D_Conv_1_1)
    Add_1 = keras.layers.add([D_Conv_1_1, ORED_P_1])
    
    D_Conv_1_2 = keras.layers.Conv2D(1, 3,padding = 'same', kernel_initializer = 'he_normal')(Add_1)
    D_Conv_1_2=keras.layers.BatchNormalization()(D_Conv_1_2)   
    #D_Conv_1_2=keras.layers.Activation('relu')(D_Conv_1_2)
    out=keras.layers.Activation('sigmoid')(D_Conv_1_2)    
    model = keras.models.Model(inputs = inputs, outputs = out)
    return model

