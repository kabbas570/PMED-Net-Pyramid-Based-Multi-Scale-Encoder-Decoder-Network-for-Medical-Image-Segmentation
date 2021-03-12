import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

from Evaluation__Metrics import Evaluation_Metrics

				 ##### Import Datasets ########
from Datasets.Data_Nuclei import DataGenT
from Datasets.Data_X_Ray import DataGenT
from Datasets.Data_Brain import DataGenT
from Datasets.Data_ISIC import DataGenT
img_T,C1T,C2T,C3T,C4T,C5T,mask_T,Y1T,Y2T,Y3T,Y4T,Y5T=DataGenT()


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

Threshold=0.5
### Testing All Models ####
model=UNet()
model.load_weights("UNet_ISIC_DIC.h5")
result_U = model.predict(C2T,batch_size=2)

result_U[np.where(result_U[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result_U,Y2T)


model=segnet()
model.load_weights("D_SegNet.h5")
result_S = model.predict(C2T,batch_size=2)
result_S[np.where(result_S[:,:,:,0]>Threshold)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result_S,Y2T)


model=fcn_8()
model.load_weights("FCN_ISIC_DIC.h5")
result_F = model.predict(C2T,batch_size=2)
result_F[np.where(result_F[:,:,:,0]>Threshold)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result_F,Y2T)


model=BCDU_net_D3()
model.load_weights("BCDU_net_ISIC_DIC.h5")
result_B = model.predict(C2T,batch_size=2)

result_B[np.where(result_B[:,:,:,0]>=Threshold)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result_B,Y2T)


model=CU_net()
model.load_weights("CU_ISIC_BIN.h5")
result = model.predict(C2T,batch_size=2)
result_C=result[1]
result_C[np.where(result_C[:,:,:,0]>Threshold)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result_C,Y2T)

model=ORED_net()
model.load_weights("ORED_Net_XRay.h5")
result_ = model.predict(C2T,batch_size=2)
result_[np.where(result_[:,:,:,0]>Threshold)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result_,Y2T)


model=MultiResUnet()
model.load_weights("MultiResUnet_XRay.h5")
result_ = model.predict(C2T,batch_size=2)
result_[np.where(result_[:,:,:,0]>Threshold)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result_,Y2T)



model1=MYMODEL1()
model2=MYMODEL2()
model3=MYMODEL3()
model4=MYMODEL4()
model5=MYMODEL5()
model6=MYMODEL6()

model1.load_weights("b1.h5")
model2.load_weights("b2.h5")
model3.load_weights("b3.h5")
model4.load_weights("b4.h5")
model5.load_weights("b5.h5")
model6.load_weights("b6.h5")


result1 = model1.predict(C5T)
result1[np.where(result1[:,:,:,0]>.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result1,Y5T)
result2 = model2.predict([result1,C4T])
result2[np.where(result2[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result2,Y4T)
result3 = model3.predict([result2,C3T])
result3[np.where(result3[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result3,Y3T)
result4 = model4.predict([result3,C2T])
result4[np.where(result4[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result4,Y2T)
result5 = model5.predict([result4,C1T],batch_size=2)
result5[np.where(result5[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result5,Y1T)
result6 = model6.predict([result5,img_T],batch_size=2)
result6[np.where(result6[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result6,mask_T)


### testing the same size without pyramid ###
from Model_same_size import MYMODEL1,MYMODEL2,MYMODEL3,MYMODEL4
model1=MYMODEL1()
model2=MYMODEL2()
model3=MYMODEL3()
model4=MYMODEL4()

model1.load_weights("XS1.h5")
model2.load_weights("XS2.h5")
model3.load_weights("XS3.h5")
model4.load_weights("XS4.h5")


result1 = model1.predict(C2T)
result1[np.where(result1[:,:,:,0]>.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result1,Y2T)
result2 = model2.predict([result1,C2T])
result2[np.where(result2[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result2,Y2T)
result3 = model3.predict([result2,C2T])
result3[np.where(result3[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result3,Y2T)
result4 = model4.predict([result3,C2T])
result4[np.where(result4[:,:,:,0]>0.5)]=1
iou,F1,Sensitivity=Evaluation_Metrics(result4,Y2T)


### Save the Results ###
P1="/home/user01/data_ssd/Abbas/NUCLEI/Ablation/s1/"
P2="/home/user01/data_ssd/Abbas/NUCLEI/Ablation/s2/"
P3="/home/user01/data_ssd/Abbas/NUCLEI/Ablation/s3/"
P4="/home/user01/data_ssd/Abbas/NUCLEI/Ablation/s4/"
P5="/home/user01/data_ssd/Abbas/NUCLEI/Ablation/s5/"
P6="/home/user01/data_ssd/Abbas/NUCLEI/Ablation/s6/"

for i in range(62):
    cv2.imwrite(os.path.join(P1 , str(i)+".png"),result1[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P2 , str(i)+".png"),result2[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P3 , str(i)+".png"),result3[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P4 , str(i)+".png"),result4[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P5 , str(i)+".png"),result5[i,:,:,0]*255)
    cv2.imwrite(os.path.join(P6 , str(i)+".png"),result6[i,:,:,0]*255)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
