import numpy as np
from sklearn.metrics import confusion_matrix
def Evaluation_Metrics(result,GT):
    Y=np.reshape(GT,(result.shape[0]*result.shape[2]*result.shape[2],1))
    Y=Y.astype(int)
    P=np.reshape(result,(result.shape[0]*result.shape[2]*result.shape[2],1))
    P=P.astype(int)
    tn, fp, fn, tp=confusion_matrix(Y, P,labels=[0,1]).ravel()
    F1=2*tp/(2*tp+fp+fn)
    iou=tp/(tp+fn+fp) 
    Sensitivity=tp/(tp+fn)
    print("IoU  is:  ",iou)
    print("F1_Score is:  ",F1) 
    print("Sensitivity  is:  ",Sensitivity)
    return iou,F1,Sensitivity

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    