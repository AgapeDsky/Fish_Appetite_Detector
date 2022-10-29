import re
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
# from tensorflow import keras

IMG_SIZE = 400
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def load_video(path, max_frames=0, resize=(1080, 1920)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

frames = load_video('Video/1 detik/Kenyang/kenyang 2.mp4',10, resize=(540,960))
print(frames.shape)
# print(frames)
cv2.imshow('frame0',frames[0])
cv2.imshow('frame1',frames[0][200:650,50:])
# cv2.imshow('frame1',frames[1])
# cv2.imshow('frame2',frames[2])
# cv2.imshow('frame3',frames[3])
# cv2.imshow('frame4',frames[4])
# cv2.imshow('frame5',frames[5])
# cv2.imshow('frame6',frames[6])
cv2.waitKey(0)
