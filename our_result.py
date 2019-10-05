import os
import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
from  segmentation_models import Nestnet, Unet, Xnet
from helper_functions import *

raw_image=cv.imread("./data_2/dataset/all/images/Norma_Old/C/01.jpg")###########Norma_Old
# raw_image=cv.imread("./data_2/dataset/all/images/DKMP/1/13.jpg")###########DKMP
# raw_image=cv.imread("./data_2/dataset/all/images/Pathology_Old/10/09.jpg")###########Pathology_Old
# raw_image=cv.imread("./data_2/dataset/all/images/Pathology_Old/26/07.jpg")###########Pathology_Old

raw_label=cv.imread("./data_2/dataset/all/labels/Norma_Old/C/01.jpg")##############Norma_Old
# raw_label=cv.imread("./data_2/dataset/all/labels/DKMP/1/13.jpg")##############DKMP
# raw_label=cv.imread("./data_2/dataset/all/labels/Pathology_Old/10/09.jpg")##############Pathology_Old
# raw_label=cv.imread("./data_2/dataset/all/labels/Pathology_Old/26/07.jpg")##############Pathology_Old

path="./segmentation_result/DLA_RDNet/Norma_Old/"##################Norma_Old
# path="./segmentation_result/DLA/DKMP/"##################DKMP
# path="./segmentation_result/DLA//Pathology_Old/"##################Pathology_Old
# path="./segmentation_result/DLA/Pathology/"##################Pathology_Old

result=cv.imread(path+"mask.png")
cv.imshow("input0",raw_image)

drawed_imge=np.copy(raw_image)
raw_label_gray1=cv.cvtColor(raw_label,cv.COLOR_BGR2GRAY)
ret1,raw_label_binary1=cv.threshold(raw_label_gray1,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
cv.imshow("input1",raw_label_binary1)
contours1,hierarchy1=cv.findContours(raw_label_binary1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(drawed_imge,contours1,-1,(0,255,0),1)

x,y,ww,hh=cv.boundingRect(contours1[0])
draw_imge=np.copy(raw_label)
draw_imge[y-20:y+hh+20,x-20:x+ww+20]=result
draw_imge_gray2=cv.cvtColor(draw_imge,cv.COLOR_BGR2GRAY)
ret2,draw_imge_binary2=cv.threshold(draw_imge_gray2,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
cv.imshow("input2",draw_imge_binary2)
contours2,hierarchy2=cv.findContours(draw_imge_binary2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(drawed_imge,contours2,-1,(0,0,255),1)


cv.imshow('get_result',drawed_imge)

# cv.imwrite(path+"raw_image.png",raw_image)
# cv.imwrite(path+"raw_label.png",raw_label)
# cv.imwrite(path+"draw_imge.png",draw_imge)
cv.imwrite(path+"drawed_imge2.png",drawed_imge)
cv.waitKey(0)
cv.destroyAllWindows()

