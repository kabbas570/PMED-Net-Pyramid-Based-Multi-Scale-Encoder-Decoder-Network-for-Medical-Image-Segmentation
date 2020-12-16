
import tensorflow as tf
import numpy as np
import cv2
import glob
mask_id = []
for infile in sorted(glob.glob(' Path to masks of training data (All Images in a single follder)')):
    mask_id.append(infile)
image_ = []
for infile in sorted(glob.glob('Path to images of training data (All Images in a single follder)')):
    image_.append(infile)
mask_V = []
for infile in sorted(glob.glob('Path to masks of validation data (All Images in a single follder)')):
    mask_V.append(infile)
image_V = []
for infile in sorted(glob.glob('Path to images of validation data (All Images in a single follder)')):
    image_V.append(infile)
mask_T = []
for infile in sorted(glob.glob('Path to masks of test data (All Images in a single follder)')):
    mask_T.append(infile)
image_T = []
for infile in sorted(glob.glob('Path to images of test data (All Images in a single follder)')):
    image_T.append(infile) 
     


height=1536
width=1536

def DataGen6():
    img_ = []
    mask_  = []
    for i in range(len(image_)//2):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

def DataGen6_():
    img_ = []
    mask_  = []
    for i in range(len(image_)//2,len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

def DataGen6__():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_





def DataGen5():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//2,width//2), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
def DataGen4():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//4,width//4), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//4,width//4), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
def DataGen3():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//8,width//8), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//8,width//8), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
def DataGen2():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//16,width//16), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//16,width//16), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
def DataGen1():
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height//32,width//32), interpolation = cv2.INTER_AREA)
        image=image/255     
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height//32,width//32), interpolation = cv2.INTER_AREA)
        mask=mask/255
        F = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(F)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
def DataGenV():
    img_ = []
    mask_  = []
    c1=[]
    c2=[]
    c3=[]
    c4=[]
    c5=[]
    y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]

    for i in range(len(image_V)):
        image = cv2.imread(image_V[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image=image/255
        cc1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        cc2 = cv2.resize(image, (height//4,width//4), interpolation = cv2.INTER_AREA)
        cc3 = cv2.resize(image, (height//8,width//8), interpolation = cv2.INTER_AREA)
        cc4 = cv2.resize(image,(height//16,width//16), interpolation = cv2.INTER_AREA)
        cc5 = cv2.resize(image, (height//32,width//32), interpolation = cv2.INTER_AREA)
        mask = cv2.imread(mask_V[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=mask/255
        yy1 = cv2.resize(mask, (height//2,width//2), interpolation = cv2.INTER_AREA)
        yy2 = cv2.resize(mask, (height//4,width//4), interpolation = cv2.INTER_AREA)
        yy3 = cv2.resize(mask, (height//8,width//8), interpolation = cv2.INTER_AREA)
        yy4 = cv2.resize(mask, (height//16,width//16), interpolation = cv2.INTER_AREA)
        yy5 = cv2.resize(mask, (height//32,width//32), interpolation = cv2.INTER_AREA)
        F = np.expand_dims(mask, axis=-1)
        yy1 = np.expand_dims(yy1, axis=-1)
        yy2 = np.expand_dims(yy2, axis=-1)
        yy3 = np.expand_dims(yy3, axis=-1)
        yy4 = np.expand_dims(yy4, axis=-1)
        yy5 = np.expand_dims(yy5, axis=-1)
        img_.append(image)
        mask_.append(F)
        c1.append(cc1)
        c2.append(cc2) 
        c3.append(cc3)       
        c4.append(cc4) 
        c5.append(cc5)  
        y1.append(yy1)
        y2.append(yy2)
        y3.append(yy3)
        y4.append(yy4)
        y5.append(yy5)  
    C1=np.array(c1)
    C2=np.array(c2)
    C3=np.array(c3)  
    C4=np.array(c4)
    C5=np.array(c5)
    Y1=np.array(y1)
    Y2=np.array(y2)
    Y3=np.array(y3)   
    Y4=np.array(y4)
    Y5=np.array(y5)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,C1,C2,C3,C4,C5,mask_,Y1,Y2,Y3,Y4,Y5
    #return C2,Y2
   
#img_V,C1V,C2V,C3V,C4V,C5V,C6V,mask_V,Y1V,Y2V,Y3V,Y4V,Y5V,Y6V=DataGenV() 
    
def DataGenT():
    img_ = []
    mask_  = []
    c1=[]
    c2=[]
    c3=[]
    c4=[]
    c5=[]

    y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]

    for i in range(len(image_T)):
        image = cv2.imread(image_T[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image=image/255
        cc1 = cv2.resize(image, (height//2,width//2), interpolation = cv2.INTER_AREA)
        cc2 = cv2.resize(image, (height//4,width//4), interpolation = cv2.INTER_AREA)
        cc3 = cv2.resize(image, (height//8,width//8), interpolation = cv2.INTER_AREA)
        cc4 = cv2.resize(image,(height//16,width//16), interpolation = cv2.INTER_AREA)
        cc5 = cv2.resize(image, (height//32,width//32), interpolation = cv2.INTER_AREA)

        mask = cv2.imread(mask_T[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        mask=mask/255
        yy1 = cv2.resize(mask, (height//2,width//2), interpolation = cv2.INTER_AREA)
        yy2 = cv2.resize(mask, (height//4,width//4), interpolation = cv2.INTER_AREA)
        yy3 = cv2.resize(mask, (height//8,width//8), interpolation = cv2.INTER_AREA)
        yy4 = cv2.resize(mask, (height//16,width//16), interpolation = cv2.INTER_AREA)
        yy5 = cv2.resize(mask, (height//32,width//32), interpolation = cv2.INTER_AREA)

        F = np.expand_dims(mask, axis=-1)
        yy1 = np.expand_dims(yy1, axis=-1)
        yy2 = np.expand_dims(yy2, axis=-1)
        yy3 = np.expand_dims(yy3, axis=-1)
        yy4 = np.expand_dims(yy4, axis=-1)
        yy5 = np.expand_dims(yy5, axis=-1)

        img_.append(image)
        mask_.append(F)
        c1.append(cc1)
        c2.append(cc2) 
        c3.append(cc3)       
        c4.append(cc4) 
        c5.append(cc5)  

        y1.append(yy1)
        y2.append(yy2)
        y3.append(yy3)
        y4.append(yy4)
        y5.append(yy5)  

    C1=np.array(c1)
    C2=np.array(c2)
    C3=np.array(c3)  
    C4=np.array(c4)
    C5=np.array(c5)

    Y1=np.array(y1)
    Y2=np.array(y2)
    Y3=np.array(y3)   
    Y4=np.array(y4)
    Y5=np.array(y5)

    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,C1,C2,C3,C4,C5,mask_,Y1,Y2,Y3,Y4,Y5
    #return C2,Y2
