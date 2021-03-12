import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import optimizers

				 ##### Import Datasets ########
from Datasets.Data_Nuclei import DataGenV,DataGen1,DataGen2,DataGen3,DataGen4,DataGen5,DataGen6
from Datasets.Data_X_Ray import DataGenV,DataGen1,DataGen2,DataGen3,DataGen4,DataGen5,DataGen6
from Datasets.Data_Brain import DataGenV,DataGen1,DataGen2,DataGen3,DataGen4,DataGen5,DataGen6
from Datasets.Data_ISIC import DataGenV,DataGen1,DataGen2,DataGen3,DataGen4,DataGen5,DataGen6

##### Models ########
from Models.U_Net import UNet
from Models.CU_Net import CU_net
from Models.FCN_8s import fcn_8
from Models.SegNet import segnet
from Models.BCDU_Net import BCDU_net_D3
from Models.Proposed_Model import MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4,MYMODEL5,MYMODEL6
from Models.Proposed_WO_Pyramid import MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4
from Models.MultiResUnet import MultiResUnet
from Models.ORED_Net import ORED_net

			#####  Loss function and Metrics#########
from Dice_Loss import dice_coef,dice_coef_loss
epochs=50

img_V,C1V,C2V,C3V,C4V,C5V,mask_V,Y1V,Y2V,Y3V,Y4V,Y5V=DataGenV()
C2,Y2=DataGen4()

Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model=UNet()
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("B_UNET.h5")

Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model=UNet()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("D_UNET.h5")
from tensorflow.keras.optimizers import Adam
model=BCDU_net_D3()
model.summary()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])    
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=50)
model.save_weights("B_BCDU.h5")

model=BCDU_net_D3()
model.summary()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])    
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=50)
model.save_weights("BCDU_BIN1.h5")

from tensorflow.keras.optimizers import Adam
Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model=CU_net()
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[dice_coef])
model.fit(C2,[Y2,Y2],validation_data=(C2V, [Y2V,Y2V]),batch_size=2,
                    epochs=epochs)
model.save_weights("B_CU.h5")
model=CU_net()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C2,[Y2,Y2],validation_data=(C2V, [Y2V,Y2V]),batch_size=2,
                    epochs=epochs)
model.save_weights("D_CU.h5")

Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model=fcn_8()
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("B_FCN.h5")

Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model=fcn_8()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("D_FCN.h5")

model=segnet()
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("SegNet_Brain_BIN.h5")

model=segnet()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("SegNet_Brain_DIC.h5")

Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model=UNet()
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("UNet_X_Brain1_BIN.h5")

Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model=UNet()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("UNet_Brain2_DIC.h5")

model=ORED_net()
#model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("ORED_Net_XRay.h5")

model=MultiResUnet()
#model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[dice_coef])
model.fit(C2,Y2,validation_data=(C2V, Y2V),batch_size=2,
                    epochs=epochs)
model.save_weights("MultiResUnet_XRay.h5")



epochs=50

  ###          The propsed model is trained in Cascaded using MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4, and MYMODEL5,MYMODEL6   ###
##       Stage_1      ##
Adam = optimizers.Adam(lr=0.0001,  beta_1=0.9, beta_2=0.99)
model1=MYMODEL1()
model1.summary()
img_V,C1V,C2V,C3V,C4V,C5V,mask_V,Y1V,Y2V,Y3V,Y4V,Y5V=DataGenV()
model1.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
C5,Y5=DataGen1()
model1.fit(C5,Y5,validation_data=(C5V, Y5V),batch_size=2,
                    epochs=epochs)
model1.save_weights("r1.h5")
result1 = model1.predict(C5) 
result1V = model1.predict(C5V)
##       Stage_2      ##
model2=MYMODEL2()
model2.summary()
model2.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
C4,Y4=DataGen2()
model2.fit([result1,C4],Y4,validation_data=([result1V,C4V], Y4V),batch_size=2,
                    epochs=epochs)
model2.save_weights("r2.h5")
result2 = model2.predict([result1,C4])
result2V = model2.predict([result1V,C4V])
##       Stage_3      ##
model3=MYMODEL3()
model3.summary()
model3.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
C3,Y3=DataGen3()
model3.fit([result2,C3],Y3,validation_data=([result2V,C3V], Y3V),batch_size=2,
                    epochs=epochs)
model3.save_weights("r3.h5")
result3 = model3.predict([result2,C3])
result3V = model3.predict([result2V,C3V])

##       Stage_4     ##
model4=MYMODEL4()
model4.summary()
model4.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
C2,Y2=DataGen4()
model4.fit([result3,C2],Y2,validation_data=([result3V,C2V], Y2V),batch_size=2,
                    epochs=epochs)
model4.save_weights("r4.h5")
result4 = model4.predict([result3,C2])
result4V = model4.predict([result3V,C2V])
##       Stage_5     ##
model5=MYMODEL5()
model5.summary()
model5.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
C1,Y1=DataGen5()
model5.fit([result4,C1],Y1,validation_data=([result4V,C1V], Y1V),batch_size=2,
                    epochs=epochs)
model5.save_weights("r5.h5")
result5 = model5.predict([result4,C1])
result5V = model5.predict([result4V,C1V])

##       Stage_6      ##
model6=MYMODEL6()
model6.summary()
model6.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
img_,mask_=DataGen6()
model6.fit([result5,img_],mask_,validation_data=([result5V,img_V], mask_V),batch_size=2, epochs=epochs)
model6.save_weights("r6.h5")











