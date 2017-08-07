from statistics import mode
from .utils import preprocess_input
from .utils import get_labels
import sys
from os.path import basename
from keras.models import load_model
import numpy as np
import cv2

EMA = 5e-1

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

# video
video_capture = cv2.VideoCapture(sys.argv[1])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = video_capture.read()
out_video = cv2.VideoWriter(basename(sys.argv[1]),fourcc, 20.0, (frame.shape[1], frame.shape[0]))


font = cv2.FONT_HERSHEY_DUPLEX
#cv2.namedWindow('window_frame')
emotion_label_window = []
gender_label_window = []
it = 0
ema_emotion = np.ones(7) / 7
while True:
    it += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = frame[(y - y_offset):(y + h + y_offset),
                    (x - x_offset):(x + w + x_offset)]

        gray_face = gray[(y - y_offset_emotion):(y + h + y_offset_emotion),
                        (x - x_offset_emotion):(x + w + x_offset_emotion)]
        try:
            face = cv2.resize(face, (48, 48))
            gray_face = cv2.resize(gray_face, (48, 48))
        except:
            continue
        face = np.expand_dims(face, 0)
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
        ema_emotion += EMA * ( emotion_label - ema_emotion )
        emotion_label_arg = np.argmax(ema_emotion)
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

        if emotion_label_arg == 0:
            emotion_color = (200, 0, 0)
        elif emotion_label_arg == 1:
            emotion_color = (200, 200, 0)
        elif emotion_label_arg == 2:
            emotion_color = (0, 80, 230)
        elif emotion_label_arg == 3:
            emotion_color = (30, 250, 150)
        elif emotion_label_arg == 4:
            emotion_color = (0, 50, 250)
        elif emotion_label_arg == 5:
            emotion_color = (0, 255, 0)
        else:
            emotion_color = (255, 255, 255)


        #cv2.rectangle(frame, (x - x_offset, y - y_offset),
                    #(x + w + x_offset, y + h + y_offset),
        cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 2)

        cv2.putText(frame, emotion_mode, (x, y - 30), font,
                        2, emotion_color, 1, cv2.LINE_AA)
        #cv2.putText(frame, gender_mode, (x, y + h), font,
        #                2, gender_color, 1, cv2.LINE_AA)
        # sexist pig
        print('%3d'%it, '[%1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f]' % tuple(ema_emotion.tolist()))

    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #cv2.imshow('window_frame', frame)
        out_video.write(frame)

    except:
        continue

    ret, frame = video_capture.read()

    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
#cv2.destroyAllWindows()
