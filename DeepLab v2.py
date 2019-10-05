
from __future__ import print_function
import warnings

warnings.filterwarnings('ignore')
import os
import keras

print("Keras = {}".format(keras.__version__))
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pylab
import sys
import math
import SimpleITK as sitk
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import shutil
from sklearn import metrics
import random
from random import shuffle
from keras.callbacks import LambdaCallback, TensorBoard
from glob import glob
from skimage.transform import resize
from optparse import OptionParser
from segmentation_models import Nestnet, Unet, Xnet
from helper_functions import *
from keras.utils import plot_model

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--run", dest="run", help="the index of gpu are used", default=16, type="int")
parser.add_option("--arch", dest="arch", help="Unet", default="DeepLab V2", type="string")
parser.add_option("--init", dest="init", help="random | finetune", default="random", type="string")
parser.add_option("--backbone", dest="backbone", help="the backbones", default="resnet50", type="string")
parser.add_option("--decoder", dest="decoder_block_type", help="transpose | upsampling", default="transpose",
                  type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=128, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=128, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=3, type="int")
parser.add_option("--nb_class", dest="nb_class", help="number of class", default=1, type="int")
parser.add_option("--verbose", dest="verbose", help="verbose", default=0, type="int")
parser.add_option("--weights", dest="weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--data", dest="DATA_DIR", help="data set location", default="./data_2/data", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=16, type="int")################

(options, args) = parser.parse_args()

assert options.backbone in ['vgg16',
                            'vgg19',
                            'resnet18',
                            'resnet34',
                            'resnet50',
                            'resnet101',
                            'resnet152',
                            'resnext50',
                            'resnext101',
                            'densenet121',
                            'densenet169',
                            'densenet201',
                            'inceptionv3',
                            'inceptionresnetv2',
                            "RDN",
                            ]
assert options.arch in ['Unet',
                        'Nestnet',
                        'Xnet',
                        'DeepLab V2'
                        ]
assert options.init in ['random',
                        'finetune',
                        ]
assert options.decoder_block_type in ['transpose',
                                      'upsampling'
                                      ]

# In[2]:


model_path_idx = options.run
model_path = "./trained_weights/DeepLab V2" + str(model_path_idx) + "/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


class setup_config():
    optimizer = "Adam"
    lr = 1e-4
    GPU_COUNT = 1
    nb_epoch = 100000
    patience = 30########################
    deep_supervision = False

    def __init__(self, model="DeepLab V2",
                 backbone="resnet50",
                 init="random",
                 data_augmentation=True,
                 input_rows=128,
                 input_cols=128,
                 input_deps=3,
                 batch_size=8,
                 verbose=1,
                 decoder_block_type=None,
                 nb_class=None,
                 DATA_DIR="./data_2/data",
                 ):
        self.model = model
        self.backbone = backbone
        self.init = init
        self.exp_name = model + "-" + backbone + "-" + init
        self.data_augmentation = data_augmentation
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.decoder_block_type = decoder_block_type
        self.nb_class = nb_class
        self.DATA_DIR = DATA_DIR
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"
        if self.init != "finetune":
            self.weights = None
        else:
            self.weights = "imagenet"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and "ids" not in a:
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = setup_config(model=options.arch,
                      backbone=options.backbone,
                      init=options.init,
                      input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      batch_size=options.batch_size,
                      verbose=options.verbose,
                      decoder_block_type=options.decoder_block_type,
                      nb_class=options.nb_class,
                      DATA_DIR=options.DATA_DIR,
                      )
config.display()

# In[3]:


x_train = np.load(os.path.join(config.DATA_DIR, "train_images.npy"))
y_train = np.load(os.path.join(config.DATA_DIR, "train_labels.npy"))
nb_cases = x_train.shape[0]
ind_list = [i for i in range(nb_cases)]
shuffle(ind_list)
nb_valid = int(nb_cases * 0.2)
x_valid, y_valid = x_train[ind_list[:nb_valid]], y_train[ind_list[:nb_valid]]
x_train, y_train = x_train[ind_list[nb_valid:]], y_train[ind_list[nb_valid:]]
# x_train, y_train = np.einsum('ijkl->iklj', x_train), np.einsum('ijkl->iklj', y_train)
# x_valid, y_valid = np.einsum('ijkl->iklj', x_valid), np.einsum('ijkl->iklj', y_valid)
y_train = np.array(y_train > 0, dtype="int")[:, :, :, 0:1]
y_valid = np.array(y_valid > 0, dtype="int")[:, :, :, 0:1]
print(">> Train data: {} | {} ~ {}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print(">> Train mask: {} | {} ~ {}\n".format(y_train.shape, np.min(y_train), np.max(y_train)))
print(">> Valid data: {} | {} ~ {}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
print(">> Valid mask: {} | {} ~ {}\n".format(y_valid.shape, np.min(y_valid), np.max(y_valid)))

x_test = np.load(os.path.join(config.DATA_DIR, "test_images.npy"))
y_test = np.load(os.path.join(config.DATA_DIR, "test_labels.npy"))
# x_test, y_test = np.einsum('ijkl->iklj', x_test), np.einsum('ijkl->iklj', y_test)
y_test = np.array(y_test > 0, dtype="int")[:, :, :, 0:1]
print(">> Test  data: {} | {} ~ {}".format(x_test.shape, np.min(x_test), np.max(x_test)))
print(">> Test  mask: {} | {} ~ {}\n".format(y_test.shape, np.min(y_test), np.max(y_test)))

# In[27]:
from keras.layers import *
def resnet50_deeplab(H=128,W=128,NUM_CLASSES=1):
    """ Building a resnet50 model fr semantic segmentation
    :return : returns thekeras implementation of deeplab architecture
    """
    ################################################
    ######## Building the model ####################
    ################################################
    input_layer = Input(shape=(None, None, 3), name='input_layer')
    conv1_1  = Conv2D(filters=64, kernel_size=7, strides=(2,2), use_bias=False, padding='same', name='conv1')(input_layer)
    bn1_1    = BatchNormalization(name='bn_conv1')(conv1_1)
    relu1_1  = Activation('relu')(bn1_1)
    mxp1_1   = MaxPooling2D(pool_size=3, strides=(2,2))(relu1_1)
    conv1_2  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2a_branch1')(mxp1_1)
    bn1_2    = BatchNormalization(name='bn2a_branch1')(conv1_2)

    conv2_1  = Conv2D(filters=64, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2a_branch2a')(mxp1_1)
    bn2_1    = BatchNormalization(name='bn2a_branch2a')(conv2_1)
    relu2_1  = Activation('relu')(bn2_1)
    conv2_2  = Conv2D(filters=64, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res2a_branch2b')(relu2_1)
    bn2_2    = BatchNormalization(name='bn2a_branch2b')(conv2_2)
    relu2_2  = Activation('relu')(bn2_2)
    conv2_3  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2a_branch2c')(relu2_2)
    bn2_3    = BatchNormalization(name='bn2a_branch2c')(conv2_3)

    merge3   = Add()([bn1_2, bn2_3])
    relu3_1  = Activation('relu')(merge3)
    conv3_1  = Conv2D(filters=64, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2b_branch2a')(relu3_1)
    bn3_1    = BatchNormalization(name='bn2b_branch2a')(conv3_1)
    relu3_2  = Activation('relu')(bn3_1)
    conv3_2  = Conv2D(filters=64, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res2b_branch2b')(relu3_2)
    bn3_2    = BatchNormalization(name='bn2b_branch2b')(conv3_2)
    relu3_3  = Activation('relu')(bn3_2)
    conv3_3  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2b_branch2c')(relu3_3)
    bn3_3    = BatchNormalization(name='bn2b_branch2c')(conv3_3)

    merge4   = Add()([relu3_1, bn3_3])
    relu4_1  = Activation('relu')(merge4)
    conv4_1  = Conv2D(filters=64, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2c_branch2a')(relu4_1)
    bn4_1    = BatchNormalization(name='bn2c_branch2a')(conv4_1)
    relu4_2  = Activation('relu')(bn4_1)
    conv4_2  = Conv2D(filters=64, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res2c_branch2b')(relu4_2)
    bn4_2    = BatchNormalization(name='bn2c_branch2b')(conv4_2)
    relu4_3  = Activation('relu')(bn4_2)
    conv4_3  = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res2c_branch2c')(relu4_3)
    bn4_3    = BatchNormalization(name='bn2c_branch2c')(conv4_3)

    merge5   = Add()([relu4_1, bn4_3])
    relu5_1  = Activation('relu')(merge5)
    conv5_1  = Conv2D(filters=512, kernel_size=1, strides=(2,2), use_bias=False, padding='same', name='res3a_branch1')(relu5_1)
    bn5_1    = BatchNormalization(name='bn3a_branch1')(conv5_1)

    conv6_1  = Conv2D(filters=128, kernel_size=1, strides=(2,2), use_bias=False, padding='same', name='res3a_branch2a')(relu5_1)
    bn6_1    = BatchNormalization(name='bn3a_branch2a')(conv6_1)
    relu6_1  = Activation('relu')(bn6_1)
    conv6_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3a_branch2b')(relu6_1)
    bn6_2    = BatchNormalization(name='bn3a_branch2b')(conv6_2)
    relu6_2  = Activation('relu')(bn6_2)
    conv6_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3a_branch2c')(relu6_2)
    bn6_3    = BatchNormalization(name='bn3a_branch2c')(conv6_3)

    merge7   = Add()([bn5_1, bn6_3])
    relu7_1  = Activation('relu', name='res3a_relu')(merge7)
    conv7_1  = Conv2D(filters=128, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b1_branch2a')(relu7_1)
    bn7_1    = BatchNormalization(name='bn3b1_branch2a')(conv7_1)
    relu7_2  = Activation('relu')(bn7_1)
    conv7_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3b1_branch2b')(relu7_2)
    bn7_2    = BatchNormalization(name='bn3b1_branch2b')(conv7_2)
    relu7_3  = Activation('relu')(bn7_2)
    conv7_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b1_branch2c')(relu7_3)
    bn7_3    = BatchNormalization(name='bn3b1_branch2c')(conv7_3)

    merge8   = Add()([relu7_1, bn7_3])
    relu8_1  = Activation('relu', name='res3b1_relu')(merge8)
    conv8_1  = Conv2D(filters=128, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b2_branch2a')(relu8_1)
    bn8_1    = BatchNormalization(name='bn3b2_branch2a')(conv8_1)
    relu8_2  = Activation('relu')(bn8_1)
    conv8_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3b2_branch2b')(relu8_2)
    bn8_2    = BatchNormalization(name='bn3b2_branch2b')(conv8_2)
    relu8_3  = Activation('relu')(bn8_2)
    conv8_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b2_branch2c')(relu8_3)
    bn8_3    = BatchNormalization(name='bn3b2_branch2c')(conv8_3)

    merge9   = Add()([relu8_1, bn8_3])
    relu9_1  = Activation('relu', name='res3b2_relu')(merge9)
    conv9_1  = Conv2D(filters=128, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b3_branch2a')(relu9_1)
    bn9_1    = BatchNormalization(name='bn3b3_branch2a')(conv9_1)
    relu9_2  = Activation('relu')(bn9_1)
    conv9_2  = Conv2D(filters=128, kernel_size=3, strides=(1,1), use_bias=False, padding='same', name='res3b3_branch2b')(relu9_2)
    bn9_2    = BatchNormalization(name='bn3b3_branch2b')(conv9_2)
    relu9_3  = Activation('relu')(bn9_2)
    conv9_3  = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res3b3_branch2c')(relu9_3)
    bn9_3    = BatchNormalization(name='bn3b3_branch2c')(conv9_3)

    merge10  = Add()([relu9_1, bn9_3])
    relu10_1 = Activation('relu', name='res3b3_relu')(merge10)
    conv10_1 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4a_branch1')(relu10_1)
    bn10_1   = BatchNormalization(name='bn4a_branch1')(conv10_1)

    conv11_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4a_branch2a')(relu10_1)
    bn11_1   = BatchNormalization(name='bn4a_branch2a')(conv11_1)
    relu11_1 = Activation('relu')(bn11_1)
    at_conv11_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4a_branch2b')(relu11_1)
    bn11_2   = BatchNormalization(name='bn4a_branch2b')(at_conv11_2)
    relu11_2 = Activation('relu')(bn11_2)
    conv11_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4a_branch2c')(relu11_2)
    bn11_3   = BatchNormalization(name='bn4a_branch2c')(conv11_3)

    merge12  = Add()([bn10_1, bn11_3])
    relu12_1 = Activation('relu', name='res4a_relu')(merge12)
    conv12_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b1_branch2a')(relu12_1)
    bn12_1   = BatchNormalization(name='bn4b1_branch2a')(conv12_1)
    relu12_2 = Activation('relu')(bn12_1)
    conv12_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b1_branch2b')(relu12_2)
    bn12_2   = BatchNormalization(name='bn4b1_branch2b')(conv12_2)
    relu12_3 = Activation('relu')(bn12_2)
    conv12_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b1_branch2c')(relu12_3)
    bn12_3   = BatchNormalization(name='bn4b1_branch2c')(conv12_3)

    merge13  = Add()([relu12_1, bn12_3])
    relu13_1 = Activation('relu', name='res4b1_relu')(merge13)
    conv13_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b2_branch2a')(relu13_1)
    bn13_1   = BatchNormalization(name='bn4b2_branch2a')(conv13_1)
    relu13_2 = Activation('relu')(bn13_1)
    conv13_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b2_branch2b')(relu13_2)
    bn13_2   = BatchNormalization(name='bn4b2_branch2b')(conv13_2)
    relu13_3 = Activation('relu')(bn13_2)
    conv13_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b2_branch2c')(relu13_3)
    bn13_3   = BatchNormalization(name='bn4b2_branch2c')(conv13_3)

    merge14  = Add()([relu13_1, bn13_3])
    relu14_1 = Activation('relu', name='res4b2_relu')(merge14)
    conv14_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b3_branch2a')(relu14_1)
    bn14_1   = BatchNormalization(name='bn4b3_branch2a')(conv14_1)
    relu14_2 = Activation('relu')(bn14_1)
    conv14_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b3_branch2b')(relu14_2)
    bn14_2   = BatchNormalization(name='bn4b3_branch2b')(conv14_2)
    relu14_3 = Activation('relu')(bn14_2)
    conv14_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b3_branch2c')(relu14_3)
    bn14_3   = BatchNormalization(name='bn4b3_branch2c')(conv14_3)

    merge15  = Add()([relu14_1, bn14_3])
    relu15_1 = Activation('relu', name='res4b3_relu')(merge15)
    conv15_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b4_branch2a')(relu15_1)
    bn15_1   = BatchNormalization(name='bn4b4_branch2a')(conv15_1)
    relu15_2 = Activation('relu')(bn15_1)
    conv15_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b4_branch2b')(relu15_2)
    bn15_2   = BatchNormalization(name='bn4b4_branch2b')(conv15_2)
    relu15_3 = Activation('relu')(bn15_2)
    conv15_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b4_branch2c')(relu15_3)
    bn15_3   = BatchNormalization(name='bn4b4_branch2c')(conv15_3)

    merge16  = Add()([relu15_1, bn15_3])
    relu16_1 = Activation('relu', name='res4b4_relu')(merge16)
    conv16_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b5_branch2a')(relu16_1)
    bn16_1   = BatchNormalization(name='bn4b5_branch2a')(conv16_1)
    relu16_2 = Activation('relu')(bn16_1)
    conv16_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b5_branch2b')(relu16_2)
    bn16_2   = BatchNormalization(name='bn4b5_branch2b')(conv16_2)
    relu16_3 = Activation('relu')(bn16_2)
    conv16_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b5_branch2c')(relu16_3)
    bn16_3   = BatchNormalization(name='bn4b5_branch2c')(conv16_3)

    merge17  = Add()([relu16_1, bn16_3])
    relu17_1 = Activation('relu', name='res4b5_relu')(merge17)
    conv17_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b6_branch2a')(relu17_1)
    bn17_1   = BatchNormalization(name='bn4b6_branch2a')(conv17_1)
    relu17_2 = Activation('relu')(bn17_1)
    conv17_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b6_branch2b')(relu17_2)
    bn17_2   = BatchNormalization(name='bn4b6_branch2b')(conv17_2)
    relu17_3 = Activation('relu')(bn17_2)
    conv17_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b6_branch2c')(relu17_3)
    bn17_3   = BatchNormalization(name='bn4b6_branch2c')(conv17_3)

    merge18  = Add()([relu17_1, bn17_3])
    relu18_1 = Activation('relu', name='res4b6_relu')(merge18)
    conv18_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b7_branch2a')(relu18_1)
    bn18_1   = BatchNormalization(name='bn4b7_branch2a')(conv18_1)
    relu18_2 = Activation('relu')(bn18_1)
    conv18_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b7_branch2b')(relu18_2)
    bn18_2   = BatchNormalization(name='bn4b7_branch2b')(conv18_2)
    relu18_3 = Activation('relu')(bn18_2)
    conv18_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b7_branch2c')(relu18_3)
    bn18_3   = BatchNormalization(name='bn4b7_branch2c')(conv18_3)

    merge19  = Add()([relu18_1, bn18_3])
    relu19_1 = Activation('relu', name='res4b7_relu')(merge19)
    conv19_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b8_branch2a')(relu19_1)
    bn19_1   = BatchNormalization(name='bn4b8_branch2a')(conv19_1)
    relu19_2 = Activation('relu')(bn19_1)
    conv19_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b8_branch2b')(relu19_2)
    bn19_2   = BatchNormalization(name='bn4b8_branch2b')(conv19_2)
    relu19_3 = Activation('relu')(bn19_2)
    conv19_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b8_branch2c')(relu19_3)
    bn19_3   = BatchNormalization(name='bn4b8_branch2c')(conv19_3)

    merge20  = Add()([relu19_1, bn19_3])
    relu20_1 = Activation('relu', name='res4b8_relu')(merge20)
    conv20_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b9_branch2a')(relu20_1)
    bn20_1   = BatchNormalization(name='bn4b9_branch2a')(conv20_1)
    relu20_2 = Activation('relu')(bn20_1)
    conv20_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b9_branch2b')(relu20_2)
    bn20_2   = BatchNormalization(name='bn4b9_branch2b')(conv20_2)
    relu20_3 = Activation('relu')(bn20_2)
    conv20_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b9_branch2c')(relu20_3)
    bn20_3   = BatchNormalization(name='bn4b9_branch2c')(conv20_3)

    merge21  = Add()([relu20_1, bn20_3])
    relu21_1 = Activation('relu', name='res4b9_relu')(merge21)
    conv21_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b10_branch2a')(relu21_1)
    bn21_1   = BatchNormalization(name='bn4b10_branch2a')(conv21_1)
    relu21_2 = Activation('relu')(bn21_1)
    conv21_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b10_branch2b')(relu21_2)
    bn21_2   = BatchNormalization(name='bn4b10_branch2b')(conv21_2)
    relu21_3 = Activation('relu')(bn21_2)
    conv21_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b10_branch2c')(relu21_3)
    bn21_3   = BatchNormalization(name='bn4b10_branch2c')(conv21_3)

    merge22  = Add()([relu21_1, bn21_3])
    relu22_1 = Activation('relu', name='res4b10_relu')(merge22)
    conv22_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b11_branch2a')(relu22_1)
    bn22_1   = BatchNormalization(name='bn4b11_branch2a')(conv22_1)
    relu22_2 = Activation('relu')(bn22_1)
    conv22_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b11_branch2b')(relu22_2)
    bn22_2   = BatchNormalization(name='bn4b11_branch2b')(conv22_2)
    relu22_3 = Activation('relu')(bn22_2)
    conv22_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b11_branch2c')(relu22_3)
    bn22_3   = BatchNormalization(name='bn4b11_branch2c')(conv22_3)

    merge23  = Add()([relu22_1, bn22_3])
    relu23_1 = Activation('relu', name='res4b11_relu')(merge23)
    conv23_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b12_branch2a')(relu23_1)
    bn23_1   = BatchNormalization(name='bn4b12_branch2a')(conv23_1)
    relu23_2 = Activation('relu')(bn23_1)
    conv23_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b12_branch2b')(relu23_2)
    bn23_2   = BatchNormalization(name='bn4b12_branch2b')(conv23_2)
    relu23_3 = Activation('relu')(bn23_2)
    conv23_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b12_branch2c')(relu23_3)
    bn23_3   = BatchNormalization(name='bn4b12_branch2c')(conv23_3)

    merge24  = Add()([relu23_1, bn23_3])
    relu24_1 = Activation('relu', name='res4b12_relu')(merge24)
    conv24_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b13_branch2a')(relu24_1)
    bn24_1   = BatchNormalization(name='bn4b13_branch2a')(conv24_1)
    relu24_2 = Activation('relu')(bn24_1)
    conv24_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b13_branch2b')(relu24_2)
    bn24_2   = BatchNormalization(name='bn4b13_branch2b')(conv24_2)
    relu24_3 = Activation('relu')(bn24_2)
    conv24_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b13_branch2c')(relu24_3)
    bn24_3   = BatchNormalization(name='bn4b13_branch2c')(conv24_3)

    merge25  = Add()([relu24_1, bn24_3])
    relu25_1 = Activation('relu', name='res4b13_relu')(merge25)
    conv25_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b14_branch2a')(relu25_1)
    bn25_1   = BatchNormalization(name='bn4b14_branch2a')(conv25_1)
    relu25_2 = Activation('relu')(bn25_1)
    conv25_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b14_branch2b')(relu25_2)
    bn25_2   = BatchNormalization(name='bn4b14_branch2b')(conv25_2)
    relu25_3 = Activation('relu')(bn25_2)
    conv25_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b14_branch2c')(relu25_3)
    bn25_3   = BatchNormalization(name='bn4b14_branch2c')(conv25_3)

    merge26  = Add()([relu25_1, bn25_3])
    relu26_1 = Activation('relu', name='res4b14_relu')(merge26)
    conv26_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b15_branch2a')(relu26_1)
    bn26_1   = BatchNormalization(name='bn4b15_branch2a')(conv26_1)
    relu26_2 = Activation('relu')(bn26_1)
    conv26_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b15_branch2b')(relu26_2)
    bn26_2   = BatchNormalization(name='bn4b15_branch2b')(conv26_2)
    relu26_3 = Activation('relu')(bn26_2)
    conv26_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b15_branch2c')(relu26_3)
    bn26_3   = BatchNormalization(name='bn4b15_branch2c')(conv26_3)

    merge27  = Add()([relu26_1, bn26_3])
    relu27_1 = Activation('relu', name='res4b15_relu')(merge27)
    conv27_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b16_branch2a')(relu27_1)
    bn27_1   = BatchNormalization(name='bn4b16_branch2a')(conv27_1)
    relu27_2 = Activation('relu')(bn27_1)
    conv27_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b16_branch2b')(relu27_2)
    bn27_2   = BatchNormalization(name='bn4b16_branch2b')(conv27_2)
    relu27_3 = Activation('relu')(bn27_2)
    conv27_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b16_branch2c')(relu27_3)
    bn27_3   = BatchNormalization(name='bn4b16_branch2c')(conv27_3)

    merge28  = Add()([relu27_1, bn27_3])
    relu28_1 = Activation('relu', name='res4b16_relu')(merge28)
    conv28_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b17_branch2a')(relu28_1)
    bn28_1   = BatchNormalization(name='bn4b17_branch2a')(conv28_1)
    relu28_2 = Activation('relu')(bn28_1)
    conv28_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b17_branch2b')(relu28_2)
    bn28_2   = BatchNormalization(name='bn4b17_branch2b')(conv28_2)
    relu28_3 = Activation('relu')(bn28_2)
    conv28_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b17_branch2c')(relu28_3)
    bn28_3   = BatchNormalization(name='bn4b17_branch2c')(conv28_3)

    merge29  = Add()([relu28_1, bn28_3])
    relu29_1 = Activation('relu', name='res4b17_relu')(merge29)
    conv29_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b18_branch2a')(relu29_1)
    bn29_1   = BatchNormalization(name='bn4b18_branch2a')(conv29_1)
    relu29_2 = Activation('relu')(bn29_1)
    conv29_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b18_branch2b')(relu29_2)
    bn29_2   = BatchNormalization(name='bn4b18_branch2b')(conv29_2)
    relu29_3 = Activation('relu')(bn29_2)
    conv29_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b18_branch2c')(relu29_3)
    bn29_3   = BatchNormalization(name='bn4b18_branch2c')(conv29_3)

    merge30  = Add()([relu29_1, bn29_3])
    relu30_1 = Activation('relu', name='res4b18_relu')(merge30)
    conv30_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b19_branch2a')(relu30_1)
    bn30_1   = BatchNormalization(name='bn4b19_branch2a')(conv30_1)
    relu30_2 = Activation('relu')(bn30_1)
    conv30_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b19_branch2b')(relu30_2)
    bn30_2   = BatchNormalization(name='bn4b19_branch2b')(conv30_2)
    relu30_3 = Activation('relu')(bn30_2)
    conv30_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b19_branch2c')(relu30_3)
    bn30_3   = BatchNormalization(name='bn4b19_branch2c')(conv30_3)

    merge31  = Add()([relu30_1, bn30_3])
    relu31_1 = Activation('relu', name='res4b19_relu')(merge31)
    conv31_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b20_branch2a')(relu31_1)
    bn31_1   = BatchNormalization(name='bn4b20_branch2a')(conv31_1)
    relu31_2 = Activation('relu')(bn31_1)
    conv31_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b20_branch2b')(relu31_2)
    bn31_2   = BatchNormalization(name='bn4b20_branch2b')(conv31_2)
    relu31_3 = Activation('relu')(bn31_2)
    conv31_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b20_branch2c')(relu31_3)
    bn31_3   = BatchNormalization(name='bn4b20_branch2c')(conv31_3)

    merge32  = Add()([relu31_1, bn31_3])
    relu32_1 = Activation('relu', name='res4b20_relu')(merge32)
    conv32_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b21_branch2a')(relu32_1)
    bn32_1   = BatchNormalization(name='bn4b21_branch2a')(conv32_1)
    relu32_2 = Activation('relu')(bn32_1)
    conv32_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b21_branch2b')(relu32_2)
    bn32_2   = BatchNormalization(name='bn4b21_branch2b')(conv32_2)
    relu32_3 = Activation('relu')(bn32_2)
    conv32_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b21_branch2c')(relu32_3)
    bn32_3   = BatchNormalization(name='bn4b21_branch2c')(conv32_3)

    merge33  = Add()([relu32_1, bn32_3])
    relu33_1 = Activation('relu', name='res4b21_relu')(merge33)
    conv33_1 = Conv2D(filters=256, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b22_branch2a')(relu33_1)
    bn33_1   = BatchNormalization(name='bn4b22_branch2a')(conv33_1)
    relu33_2 = Activation('relu')(bn33_1)
    conv33_2 = Conv2D(filters=256, kernel_size=3, dilation_rate=(2,2), use_bias=False, padding='same', name='res4b22_branch2b')(relu33_2)
    bn33_2   = BatchNormalization(name='bn4b22_branch2b')(conv33_2)
    relu33_3 = Activation('relu')(bn33_2)
    conv33_3 = Conv2D(filters=1024, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res4b22_branch2c')(relu33_3)
    bn33_3   = BatchNormalization(name='bn4b22_branch2c')(conv33_3)

    merge34  = Add()([relu33_1, bn33_3])
    relu34_1 = Activation('relu', name='res4b22_relu')(merge34)
    conv34_1 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5a_branch1')(relu34_1)
    bn34_1   = BatchNormalization(name='bn5a_branch1')(conv34_1)

    conv35_1 = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5a_branch2a')(relu34_1)
    bn35_1   = BatchNormalization(name='bn5a_branch2a')(conv35_1)
    relu35_2 = Activation('relu')(bn35_1)
    conv35_2 = Conv2D(filters=512, kernel_size=3, dilation_rate=(4,4), use_bias=False, padding='same', name='res5a_branch2b')(relu35_2)
    bn35_2   = BatchNormalization(name='bn5a_branch2b')(conv35_2)
    relu35_3 = Activation('relu')(bn35_2)
    conv35_3 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5a_branch2c')(relu35_3)
    bn35_3   = BatchNormalization(name='bn5a_branch2c')(conv35_3)

    merge36  = Add()([bn34_1, bn35_3])
    relu36_1 = Activation('relu', name='res5a_relu')(merge36)
    conv36_1 = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5b_branch2a')(relu36_1)
    bn36_1   = BatchNormalization(name='bn5b_branch2a')(conv36_1)
    relu36_2 = Activation('relu')(bn36_1)
    conv36_2 = Conv2D(filters=512, kernel_size=3, dilation_rate=(4,4), use_bias=False, padding='same', name='res5b_branch2b')(relu36_2)
    bn36_2   = BatchNormalization(name='bn5b_branch2b')(conv36_2)
    relu36_3 = Activation('relu')(bn36_2)
    conv36_3 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5b_branch2c')(relu36_3)
    bn36_3   = BatchNormalization(name='bn5b_branch2c')(conv36_3)

    merge37  = Add()([relu36_1, bn36_3])
    relu37_1 = Activation('relu', name='res5b_relu')(merge37)
    conv37_1 = Conv2D(filters=512, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5c_branch2a')(relu37_1)
    bn37_1   = BatchNormalization(name='bn5c_branch2a')(conv37_1)
    relu37_2 = Activation('relu')(bn37_1)
    conv37_2 = Conv2D(filters=512, kernel_size=3, dilation_rate=(4,4), use_bias=False, padding='same', name='res5c_branch2b')(relu37_2)
    bn37_2   = BatchNormalization(name='bn5c_branch2b')(conv37_2)
    relu37_3 = Activation('relu')(bn37_2)
    conv37_3 = Conv2D(filters=2048, kernel_size=1, strides=(1,1), use_bias=False, padding='same', name='res5c_branch2c')(relu37_3)
    bn37_3   = BatchNormalization(name='bn5c_branch2c')(conv37_3)

    merge38  = Add()([relu37_1, bn37_3])
    relu38_1 = Activation('relu', name='res5c_relu')(merge38)
    conv38_1 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(6,6), padding='same', name='fc1_voc12_c0')(relu38_1)
    conv38_2 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(12,12), padding='same', name='fc1_voc12_c1')(relu38_1)
    conv38_3 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(18,18), padding='same', name='fc1_voc12_c2')(relu38_1)
    conv38_4 = Conv2D(filters=NUM_CLASSES, kernel_size=3, dilation_rate=(24,24), padding='same', name='fc1_voc12_c3')(relu38_1)

    output   = Add(name='fc1_voc12')([conv38_1, conv38_2, conv38_3, conv38_4])
    output   = Lambda(lambda image: tf.image.resize_images(image, (H,W)))(output)
    output = Activation('sigmoid')(output)
    #output = UpSampling2D((3,3))(output)
    #output = MaxPooling2D(pool_size=(2,2), strides=(2,2))(output)
    #output   = Activation('softmax')(output)

    model = Model(inputs=input_layer, outputs=output)
    # model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
model=resnet50_deeplab()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=bce_dice_loss,
              metrics=["binary_crossentropy", mean_iou, dice_coef])


# model.compile(optimizer="Adam",
#               loss=bce_dice_loss,
#               metrics=["binary_crossentropy", mean_iou, dice_coef])

# plot_model(model, to_file=os.path.join(model_path, config.exp_name+".png"))
if os.path.exists(os.path.join(model_path, config.exp_name + ".txt")):
    os.remove(os.path.join(model_path, config.exp_name + ".txt"))
with open(os.path.join(model_path, config.exp_name + ".txt"), 'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(logs_path, config.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(logs_path, config.exp_name)):
    os.makedirs(os.path.join(logs_path, config.exp_name))
tbCallBack = TensorBoard(log_dir=os.path.join(logs_path, config.exp_name),
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True,
                         )
tbCallBack.set_model(model)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=config.patience,
                                               verbose=0,
                                               mode='min',
                                               )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, config.exp_name + ".h5"),
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              mode='min',
                                              )
callbacks = [check_point, early_stopping, tbCallBack]

while config.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        model.fit(x_train, y_train,
                  batch_size=config.batch_size,
                  epochs=config.nb_epoch,
                  verbose=config.verbose,
                  shuffle=True,
                  validation_data=(x_valid, y_valid),
                  callbacks=callbacks)
        break
    except tf.errors.ResourceExhaustedError as e:
        config.batch_size = int(config.batch_size / 2.0)
        print("\n> Batch size = {}".format(config.batch_size))


model.load_weights(os.path.join(model_path, config.exp_name + ".h5"))
model.compile(optimizer="Adam",
              loss=dice_coef_loss,
              metrics=["binary_crossentropy", mean_iou, dice_coef])
p_test = model.predict(x_test, batch_size=config.batch_size, verbose=config.verbose)
eva = model.evaluate(x_test, y_test, batch_size=config.batch_size, verbose=config.verbose)
IoU = compute_iou(y_test, p_test)
print("\nSetup: {}".format(config.exp_name))
print(">> Testing dataset mIoU  = {:.2f}%".format(np.mean(IoU)))
print(">> Testing dataset mDice = {:.2f}%".format(eva[3] * 100.0))
