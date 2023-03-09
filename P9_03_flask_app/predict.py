

import tensorflow
from tensorflow.python.keras.models import Model, load_model
import os
import sys
import numpy as np
from dilatednet_hd import DilatedNet
from multiclassunet import Unet
import cv2
from PIL import Image

dirname = os.path.dirname(__file__)
name = 'dilated.h5'
modelname = os.path.join(dirname, name)
print(modelname)
model = DilatedNet(256, 256, 8,use_ctx_module=True, bn=True)
model.load_weights(modelname)

def preprocess(img):
    img = Image.open(img)
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return img

def outprocess(pred):
    pred = np.squeeze(pred)
    pred = pred.reshape(256, 256, 8)
    pred = cv2.resize(pred, (256, 256))
    pred = np.argmax(pred, axis=2)
    return pred

def make_prediction(img):
    img = preprocess(img)
    #make prediction
    pred = model.predict(np.expand_dims(img, axis=0))
    print('prediction ok')
    mask = outprocess(pred)
    return mask

#if __name__ == '__main__':
#    make_prediction(sys.argv[1])
