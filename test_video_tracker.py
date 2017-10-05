#!/usr/bin/python
#coding:utf-8

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
    print("Usage: python camera_test.py <video source> <scale size>")
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

#Variables holding the correlation trackers and the name per faceid
frameCounter = 0
currentFaceID = 0
faceTrackers = {}
faceNames = {}
faceTrackerQualities={}
faceDetectedCounter=0

video_capture = cv2.VideoCapture(videopath)
corpbbox = None
while True:
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
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

 
    #Update all the trackers and remove the ones for which the update
    #indicated the quality was not good enough
    fidsToDelete = []
    for fid in faceTrackers.keys():
        trackingQuality = faceTrackers[ fid ].update(frame)
        faceTrackerQualities[fid][0:4]=faceTrackerQualities[fid][1:5]
        faceTrackerQualities[fid][4] = trackingQuality 

        #If the tracking quality is not good, delete it
        if sum(faceTrackerQualities[fid]>=6)<1: # if there are >=3 good quality in 5 tracker qualities 
            fidsToDelete.append( fid )            
        print("Tracker " + str(fid) + " qualities = " + str(faceTrackerQualities[fid])) 

    for fid in fidsToDelete:
        print("Removing fid " + str(fid) + " from list of trackers")
        faceTrackers.pop( fid , None )
    
    if (frameCounter % 5) == 0:

       #write face into file  
        #fn = "frame-"+str(frameCounter)+".jpg" 
        #cv2.imwrite(fn, image_orig) 

        #t0 = time.time()
        img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)    
        bounding_boxes, points, scores = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        #t = time.time()-t0 

        nrof_faces = bounding_boxes.shape[0]
        print("At frame {} Number of faces detected: {}, time = {}s".format(frameCounter,nrof_faces,t))
        faceCounter = faceCounter+nrof_faces 

       
        i=0 
        for b in bounding_boxes:
            corpbbox = [int(b[0]/scale), int(b[1]/scale), int(b[2]/scale), int(b[3]/scale)]
            cv2.rectangle(frame, (int(corpbbox [0]), int(corpbbox [1])), (int(corpbbox [2]), int(corpbbox [3])), (0, 255, 0))
            #print(b) #location of detected face 
            score = scores[i] 
            #write face into file  
            face_crop = image_orig[corpbbox[1]:corpbbox[3],corpbbox[0]:corpbbox[2]]
            #fn = "frame-"+str(frameCounter)+"-face-"+str(i)+ ".jpg" 
            #cv2.imwrite(fn, face_crop) 
            i = i+1 
            x = corpbbox[0] 
            y = corpbbox[1]
            w = corpbbox[2]-x
            h = corpbbox[3]-y

            #calculate the centerpoint
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            #Variable holding information which faceid we matched with
            matchedFid = None
            #Now loop over all the trackers and check if the centerpoint of the face is within the box of a tracker
            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                #calculate the centerpoint
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                #check if the centerpoint of the face is within the 
                #rectangleof a tracker region. Also, the centerpoint
                #of the tracker region must be within the region 
                #detected as a face. If both of these conditions hold
                #we have a match
                if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                      ( t_y <= y_bar   <= (t_y + t_h)) and 
                      ( x   <= t_x_bar <= (x   + w  )) and 
                      ( y   <= t_y_bar <= (y   + h  ))):
                    matchedFid = fid
                    #update the face in tracker to current face 
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, dlib.rectangle( x-10,y-10,x+w+10,y+h+10)) #tracker on original image 
                    faceTrackers[fid] = tracker
                    
            #If no matched fid, then we have to create a new tracker
            if matchedFid is None:
                print("Creating new tracker " + str(currentFaceID))
                #Create and store the tracker 
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, dlib.rectangle( x-10,y-10,x+w+10,y+h+10)) #tracker on original image 
                matchedFid =  currentFaceID 
                faceTrackers[currentFaceID] = tracker
                tens = np.empty(5); tens.fill(20); 
                faceTrackerQualities[ currentFaceID ] = tens  
                #pseudo face recognition
                faceNames[currentFaceID] = "Person-" + str(currentFaceID)
                #Increase the currentFaceID counter
                currentFaceID += 1
                
           
            #write face into file  
            fn = faceNames[matchedFid]+"-score-" + str(score) +"-frame-"+str(frameCounter)+".jpg" 
            cv2.imwrite(fn, face_crop) 

        
        if len(points)>0:
            for p in points.T:
                for i in range(5):
                    cv2.circle(frame, (int(p[i]/scale), int(p[i + 5]/scale)), 1, (0, 0, 255), 2)
 

    #Now loop over all the trackers we have and draw the rectangle
    #around the detected faces. If we 'know' the name for this person
    #(i.e. the recognition thread is finished), we print the name
    #of the person, otherwise the message indicating we are detecting
    #the name of the person
    for fid in faceTrackers.keys():
        tracked_position =  faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        cv2.rectangle(frame, (t_x, t_y),(t_x + t_w , t_y + t_h),(0,255,255),2)
        if fid in faceNames.keys():
            cv2.putText(frame, faceNames[fid] , 
                (int(t_x + t_w/2), int(t_y)), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Detecting..." , 
                (int(t_x + t_w/2), int(t_y)), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
   
    t2 = cv2.getTickCount()
    t = (t2 - t1) / cv2.getTickFrequency()
    fps = 1.0 / t
    cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)

    cv2.imshow("", frame)

    frameCounter += 1 
    timeCounter = timeCounter + t 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print("Total faces detected = {}, Avg detection time = {}s".format(faceCounter,timeCounter/frameCounter))

video_capture.release()
cv2.destroyAllWindows()
