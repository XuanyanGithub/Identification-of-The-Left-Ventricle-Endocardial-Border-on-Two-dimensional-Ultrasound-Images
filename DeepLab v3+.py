
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
parser.add_option("--arch", dest="arch", help="Unet", default="DeepLab V3+", type="string")
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
                        'DeepLab V3+'
                        ]
assert options.init in ['random',
                        'finetune',
                        ]
assert options.decoder_block_type in ['transpose',
                                      'upsampling'
                                      ]

# In[2]:


model_path_idx = options.run
model_path = "./trained_weights/DeepLab V3+" + str(model_path_idx) + "/"
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

    def __init__(self, model="DeepLab V3+",
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
import keras.backend as K
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils
from keras.backend.common import normalize_data_format

class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def xception_downsample_block(x, channels, top_relu=False):
    ##separable conv1
    if top_relu:
        x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv2
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_downsample_block(x, channels):
    res = Conv2D(channels, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = xception_downsample_block(x, channels)
    x = add([x, res])
    return x


def xception_block(x, channels):
    ##separable conv1
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv2
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv3
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_block(x, channels):
    res = x
    x = xception_block(x, channels)
    x = add([x, res])
    return x


def aspp(x, input_shape, out_stride):
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x


def deeplabv3_plus(input_shape=(128,128, 3), out_stride=16, num_classes=21):
    img_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = res_xception_downsample_block(x, 128)

    res = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    skip = BatchNormalization()(x)
    x = Activation("relu")(skip)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = xception_downsample_block(x, 728, top_relu=True)

    for i in range(16):
        x = res_xception_block(x, 728)

    res = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(728, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(2048, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # aspp
    x = aspp(x, input_shape, out_stride)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.9)(x)

    ##decoder
    x = BilinearUpsampling((4, 4))(x)
    dec_skip = Conv2D(48, (1, 1), padding="same", use_bias=False)(skip)
    dec_skip = BatchNormalization()(dec_skip)
    dec_skip = Activation("relu")(dec_skip)
    x = Concatenate()([x, dec_skip])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_classes, (1, 1), padding="same")(x)
    x = Activation('sigmoid')(x)
    x = BilinearUpsampling((4, 4))(x)
    model = Model(img_input, x)
    return model
model=deeplabv3_plus(num_classes=1)
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
