from PIL import Image
from statistics import mode
from emo_utils import preprocess_input
from emo_utils import get_labels
import sys
from os.path import basename
from keras.models import load_model
import numpy as np
import cv2

EMA = 9e-1
STYLE_DOWNSCALE = 1
LABEL = True
### KERAS MUST BE IMPORTED FIRST OR ELSE:
# it will throw shits like this:
# *** Error in `python': free(): invalid pointer: 0x00007f6af7b5eac0 ***
# EMOTION
############################################################

# Proof-of-concept
import cv2
import argparse
import os
import sys
import time
import glob
import scipy.signal as ss
import numpy as np
import torch

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import FloatTensor
from os.path import basename
import style_utils
from transformer_net import TransformerNet
from vgg import Vgg16
#
# STYLE
####################################################xx

def hamingweightwindow2d(width):
    hm_len = width
    bw2d = np.outer(ss.hamming(hm_len), np.ones(hm_len))
    bw2d = np.sqrt(bw2d * bw2d.T)
    return FloatTensor(bw2d).view(1, 1, width, width).cuda()

def gaussianweightwindow2d(width):
    bw2d = np.outer(cv2.getGaussianKernel(width, 0), np.ones(width))
    bw2d = np.sqrt(bw2d * bw2d.T)
    return FloatTensor(bw2d).view(1, 1, width, width).cuda()

def weightwindow2d(width):
    return hamingweightwindow2d(width)

print('LOADING MODELS...')
# parameters
detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/emotion_models/simple_CNN.530-0.65.hdf5'
gender_model_path = 'trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
frame_window = 10
x_offset_emotion = 20
y_offset_emotion = 40
x_offset = 30
y_offset = 60
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
gender_classifier = load_model(gender_model_path)
print('DETECTION MODELS ARE COMPILED!')
# video
video_capture = cv2.VideoCapture(sys.argv[1])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
W = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) / STYLE_DOWNSCALE)
H = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / STYLE_DOWNSCALE)
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
out_video = cv2.VideoWriter(basename(sys.argv[1]), fourcc, 20.0, (W, H))

font = cv2.FONT_HERSHEY_SIMPLEX

emotion_label_window = []
gender_label_window = []
it = 0

# INITIALIZE EMOTION RECOGNITION
###########################################################################
# INITIALIZE STYLE TRANSFER
style_model_paths = glob.glob('trained_models/**.pth')
style_models = []
for path in style_model_paths:
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(path))
    style_model.cuda()
    style_models.append(style_model)

def load_net(name):
    net = TransformerNet()
    net.load_state_dict(torch.load('trained_models/'+name))
    net.cuda()
    return net

rain_princess = load_net('rain_princess.pth')
udnie = load_net('udnie.pth')
candy = load_net('candy.pth')
mosaic = load_net('mosaic.pth')

style_models = {
    'neutral': udnie,
    'angry': candy,
    'disgust': udnie,
    'fear': candy,
    'sad': rain_princess,
    'happy': mosaic,
    'surprise': rain_princess
}

def plot_boxes(frame, face_coords, label_prob):


    for (x, y, h, w), emotion_label in zip(faces, label_prob):
        emotion_label_arg = np.argmax(emotion_label)
        activity = np.max(emotion_label) * 100
        emotion_probability = np.max(emotion_label)
        emotion_text = emotion_mode
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int).tolist()

        emotion_color = color

        x, y, h, w = map(lambda x: x/STYLE_DOWNSCALE, [x, y, h, w])
        x, y, h, w = map(int, [x, y, h, w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 1)
        displacement = int(18/STYLE_DOWNSCALE)
        cv2.putText(
            frame, emotion_mode + ' %2.0f'%activity,
            (x + displacement, y - displacement),
            font, 1/STYLE_DOWNSCALE,
            emotion_color, 1, cv2.LINE_AA)


print('STYLE TRANSFER MODELS ARE COMPILED!')
###########################################################################
# START PROCESSING

ema_emotion = torch.ones(7) / 7
while True:
    t = time.time()
    ret, orig_frame = video_capture.read()
    it += 1

    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame = orig_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detection.detectMultiScale(gray, 1.3, 5)
    emotion_mode = 'neutral' # Set default value

    emotion_preds = []
    for (x, y, w, h) in faces:
        face = frame[(y - y_offset):(y + h + y_offset),
                     (x - x_offset):(x + w + x_offset)]

        gray_face = gray[(y - y_offset_emotion):(y + h + y_offset_emotion),
                         (x - x_offset_emotion):(x + w + x_offset_emotion)]
        try:
            face = cv2.resize(face, (48, 48))
            gray_face = cv2.resize(gray_face, (48, 48))
            face = np.expand_dims(face, 0)
        except:
            continue
        face = preprocess_input(face)
        # gender_label_arg = np.argmax(gender_classifier.predict(face))
        # Override for presentation -> TOTALLY EXCLUDE FEATURE, its 2017!
        gender_label_arg = 0
        gender = gender_labels[gender_label_arg]
        gender_label_window.append(gender)

        gray_face = preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label = emotion_classifier.predict(gray_face)[0]
        emotion_label_arg = np.argmax(emotion_label)
        '''
        print(emotion_preds)
        print(emotion_labels)
        print(emotion_label)
        print(emotion_label_arg)'''
        emotion = emotion_labels[emotion_label_arg]
        emotion_label_window.append(emotion)
        if len(gender_label_window) >= frame_window:
            emotion_label_window.pop(0)
            gender_label_window.pop(0)
        try:
            emotion_mode = mode(emotion_label_window)
            gender_mode = mode(gender_label_window)
        except:
            continue

        emotion_preds.append(emotion_label)
    emotion_preds = np.array(emotion_preds)
    # Smoothen the values
    if emotion_preds.size != 0:
        mean_emotion = torch.mean(FloatTensor(emotion_preds))
        ema_emotion += EMA * (mean_emotion - ema_emotion)


    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #cv2.imshow('window_frame', frame)
    #out_video.write(frame)
    frame = orig_frame
    frame = style_utils.preprocess_image(frame, scale=STYLE_DOWNSCALE)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.div(255.)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x.mul(255))
        #transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    frame = content_transform(frame)
    frame = frame.unsqueeze(0)
    frame = Variable(frame, volatile=True).cuda()


    t1 = time.time()
    # frame_nostyle = frame

    #for f in [frame_nostyle, frame_udnie, frame_rain_princess,
    #          frame_mosaic, frame_candy]:
    #    print(frame.min(), frame.max())
    '''
    frame_udnie = udnie(frame)
    frame_rain_princess = rain_princess(frame)
    frame_mosaic = mosaic(frame)
    frame_candy = candy(frame)
    eem_gpu = ema_emotion.cuda()
    frame = eem_gpu[0] * frame_rain_princess
    frame += eem_gpu[1] * frame_candy
    frame += eem_gpu[2] * frame_udnie
    frame += eem_gpu[3] * frame_candy
    frame += eem_gpu[4] * frame_rain_princess
    frame += eem_gpu[5] * frame_mosaic
    frame += eem_gpu[6] * frame_mosaic
    '''
    # Frame is originally between -255 255 before inference
    frame += 255.
    frame /= 2

    for (y, x, h, w), emotion_label in zip(faces, emotion_preds):
        # x y are somehow twisted by cv2
        emotion_label_arg = np.argmax(emotion_label)
        emotion = emotion_labels[emotion_label_arg]
        style_model = style_models[emotion]
        #x -= min([max([w//5, 0]), frame.size(2)])
        #y -= min([max([h//5, 0]), frame.size(3)])
        #h += h // 5
        #w += w // 5
        style = style_model(frame[:, :, x:x+w, y:y+h])

        #weight = weightwindow2d(style.size(2)).expand_as(style)

        #style = style.data * weight +\
        #    frame[:, :, x:x+style.size(2), y:y+style.size(3)].data *\
        #    (torch.ones(weight.size()).cuda() - weight)
        #print(frame.std(), frame.min(), frame.max())
        frame[:, :, x:x+style.size(2), y:y+style.size(3)] = style
        #print(frame.std(), frame.min(), frame.max())


    frame = style_utils.postprocess_image(frame.cpu().data.squeeze())
    frame = frame.astype(np.uint8).copy()

    if LABEL and len(faces) > 0 and emotion_preds.size > 0:  # LABEL the choice
        plot_boxes(frame, faces, emotion_preds)

    #print(frame.min(), frame.max(), frame.std(), frame.shape)

    out_video.write(frame)
    print('[ %4d / %4d ] FPS: %5.2f, FPSface %5.2f' %\
        (it, length, 1/(time.time()-t), 1/(time.time()-t1)))
        #'[%3d, %3d, %3d, %3d, %3d, %3d, %3d]' %
        #tuple((ema_emotion*100).astype('int').tolist()))
out_video.release()
video_capture.release()
print("Ended!")

'''
if it == 1:
    frame_ema = frame
else:
    frame_ema += EMA * (frame - frame_ema)
'''
