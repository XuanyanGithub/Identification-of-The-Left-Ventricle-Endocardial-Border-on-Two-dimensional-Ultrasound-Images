import os
import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
from  segmentation_models import Nestnet, Unet, Xnet,PSPNet,FPN
from helper_functions import *
from time import time

def compute_APD_GC(img1, img2):
    img1 = np.array(img1)
    img1=np.uint8(img1*255)
    img2 = np.array(img2)
    img2 = np.uint8(img2*255)
    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Shape mismatch: the number of images mismatch.")
    APD = np.zeros( (img1.shape[0],), dtype=np.float32)
    k=0
    for i in range(img1.shape[0]):
        #gray1=cv.cvtColor(img1[i],cv.COLOR_BGR2GRAY)
        ret1,binary1=cv.threshold(img1[i],0,1,cv.THRESH_BINARY|cv.THRESH_OTSU)
        contours1,hierachy1=cv.findContours(binary1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if type(contours1) is list:
            contours1=contours1[0]
        mm=cv.moments(contours1)
        m00=mm['m00']
        m10=mm['m10']
        m01=mm['m01']
        if m00==0:
            m00=0.0000000000000001
        cx1=np.int(m10/m00)
        cy1=np.int(m01/m00)

        # gray2=cv.cvtColor(img2[i],cv.COLOR_BGR2GRAY)
        ret2,binary2=cv.threshold(img2[i],0,1,cv.THRESH_BINARY|cv.THRESH_OTSU)
        contours2,hierachy2=cv.findContours(binary2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        if type(contours2) is list:
            contours2=contours2[0]
        mm = cv.moments(contours2)
        m00 = mm['m00']
        m10 = mm['m10']
        m01 = mm['m01']
        if m00 == 0:
            m00 = 0.000000000000001
        cx2 = np.int(m10 / m00)
        cy2= np.int(m01 / m00)
        APD[i]=np.sqrt((cx2-cx1)**2+(cy2-cy1)**2)/5.0#####
        if APD[i]<5:
            k+=1
    A=np.mean(APD)
    GC=k/img1.shape[0]
    return A,GC

model_path = "./trained_weights/DLA_16/4/"###########

result="./segmentation_result/DLA/1/Norma_Old/"############Norma_Old
# result="./segmentation_result/DLA/1/DKMP/"############DKMP
# result="./segmentation_result/DLA/1/Pathology_Old/"############Pathology_Old
# result="./segmentation_result/DLA/2/Pathology/"############Pathology

x_test = np.load(os.path.join("./data_2/data", "test_images.npy"))
y_test = np.load(os.path.join("./data_2/data", "test_labels.npy"))
y_test = np.array(y_test>0, dtype="int")[:,:,:,0:1]
nb_cases = x_test.shape[0]
ind_list = [i for i in range(nb_cases)]

x_test_1 = x_test[ind_list[0]]################Norma_Old
# x_test_1 = x_test[ind_list[120]]################DKMP
# x_test_1 = x_test[ind_list[256]]################Pathology_Old
# x_test_1 = x_test[ind_list[337]]################Pathology

x_test_1 = np.expand_dims(x_test_1,axis=0)

y_test_1 = y_test[ind_list[0]]##############Norma_Old
# y_test_1 = y_test[ind_list[120]]##############DKMP
# y_test_1 = y_test[ind_list[256]]##############Pathology_Old
# y_test_1 = y_test[ind_list[337]]##############Pathology

y_test_1 = np.expand_dims(y_test_1,axis=0)
y_test_1 = np.array(y_test_1>0, dtype="int")[:,:,:,0:1]


start_time=time()
model = Nestnet(backbone_name="RDN",###########################
                encoder_weights= None,
                n_upsample_blocks=4,
                decoder_block_type='transpose',
                classes=1,
                activation="sigmoid")

model.load_weights(os.path.join(model_path,"Nestnet-RDN-random.h5"))################
model.compile(optimizer="Adam",
              loss=dice_coef_loss,
              metrics=["binary_crossentropy", mean_iou, dice_coef])
p_test = model.predict(x_test, batch_size=32, verbose=1)
model.summary()
print(">> Testing dataset time:",time()-start_time)
eva = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
IoU = compute_iou(y_test, p_test)
mAPD,GC = compute_APD_GC(y_test, p_test)
print(">> Testing dataset mIoU  = {:.2f}%".format(np.mean(IoU)))
print(">> Testing dataset mDice = {:.2f}%".format(eva[3]*100.0))
print(">> Testing dataset mAPD = {:.2f}".format(mAPD))
print(">> Testing dataset GC = {:.2f}%".format(GC*100.0))


p1_test = model.predict(x_test_1)
IoU_1=compute_iou(y_test_1,p1_test)
print(">> Testing one dataset mIoU  = {}%".format(IoU_1))

src=cv.imread("./data_2/dataset_1/labels/Norma_Old/C/01.jpg")#############Norma_Old
# src=cv.imread("./data_2/dataset_1/labels/DKMP/1/13.jpg")#############DKMP
# src=cv.imread("./data_2/dataset_1/labels/Pathology_Old/10/09.jpg")#############Pathology_Old
# src=cv.imread("./data_2/dataset_1/labels/Pathology_Old/26/07.jpg")#############Pathology

h,w = src.shape[:2]
cv.imshow("input",src)
im=np.array([x for y in p1_test for x in y])#jiang wei
output=cv.resize(im,(w,h),interpolation=cv.INTER_LINEAR)
cv.imshow("output",output)
if not os.path.exists(result):
    os.makedirs(result)
dst=((output-output.min())/(output.max()-output.min()))*255
cv.imwrite(result+"mask.png",dst)
cv.waitKey(0)
cv.destroyAllWindows()

