from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
# import playsound
import argparse
import imutils
import time
import dlib
import cv2
import subprocess

def sound_alarm(path):
	subprocess.Popen(['mpg123', '-q', path]).wait()

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
 
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
h = int(cap.get(3))
w = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('recording.avi',fourcc,30,(int(cap.get(3)),int(cap.get(4))))
time.sleep(1.0)

while True:

	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True
					t = Thread(target=sound_alarm,
						args=('beep.mp3',))
					t.deamon = True
					t.start()

				cv2.putText(frame, "DROWSINESS WARNING!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			COUNTER = 0
			ALARM_ON = False

	cv2.imshow("Eyeframe", frame)
	# out.write(frame)

	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
cap.release()