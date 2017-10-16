# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# Borrowed from davidsandberg's facenet project: https://github.com/davidsandberg/facenet
# From this directory:
#   facenet/src/align
#
# Just keep the MTCNN related stuff and removed other codes
# python package required:
#     tensorflow, opencv,numpy


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import dlib

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
import time

if len(sys.argv) !=3:
    print("Usage: python test_video.py <video source> <scale size>")
    print(" scale should be 0.1 to 1")
    sys.exit()

videopath = sys.argv[1]

try:
    scale = float(sys.argv[2])
except:
    print("Scale size is not float.")
    sys.exit()

if scale > 1 or scale < 0.1:
    print("Scale is out of range, use 0.5 as default")
    scale = 0.5

t0 = time.time()

sess = tf.Session()
pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
factor = 0.309 # scale factor

video_capture = cv2.VideoCapture(videopath)
corpbbox = None
frameCounter = 0
faceCounter = 0
timeCounter = 0

t = time.time()-t0
print("Loading time = {}s".format(t))
dlib.hit_enter_to_continue()


while True:

    ret, frame = video_capture.read()
    if not ret:
        print("device not find")
        break

    height, width, channels = frame.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    frame_resized = cv2.resize(frame, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    image = np.array(frame_resized)
    image_orig = np.array(frame)

    t0 = time.time()
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    bounding_boxes, points scores = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    t = time.time()-t0

    nrof_faces = bounding_boxes.shape[0]
    print("At frame {} Number of faces detected: {}, time = {}s".format(frameCounter,nrof_faces,t))
    faceCounter = faceCounter+nrof_faces
    timeCounter = timeCounter + t

    i=0
    for b in bounding_boxes:
        corpbbox = [int(b[0]/scale), int(b[1]/scale), int(b[2]/scale), int(b[3]/scale)]
        cv2.rectangle(frame, (int(corpbbox [0]), int(corpbbox [1])), (int(corpbbox [2]), int(corpbbox [3])), (0, 255, 0))
        print(b)
        #write face into file
        face_crop = image_orig[corpbbox[1]:corpbbox[3],corpbbox[0]:corpbbox[2]]
        fn = "frame-"+str(frameCounter)+"-face-"+str(i)+ ".jpg"
        cv2.imwrite(fn, face_crop)
        i = i+1

    if len(points)>0:
        for p in points.T:
            for i in range(5):
                cv2.circle(frame, (int(p[i]/scale), int(p[i + 5]/scale)), 1, (0, 0, 255), 2)

    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameCounter = frameCounter+1

print("Total faces detected = {}, Avg detection time = {}s".format(faceCounter,timeCounter/frameCounter))

video_capture.release()
cv2.destroyAllWindows()


