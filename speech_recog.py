import time

import cv2
import numpy as np
import speech_recognition as sr
#
# class SpeechRecognition():

mic_name = "HDA Intel PCH: ALC3227 Analog (hw:0,0)"
sample_rate = 44100
chunk_size = 1024
mic_list = sr.Microphone.list_microphone_names()
device_id=1
for i, microphone_name in enumerate(mic_list):
   if microphone_name == mic_name:
       device_id = i

def recog(source):
   r.adjust_for_ambient_noise(source)
   print("Say Something...")
   audio = r.listen(source,timeout=3,phrase_time_limit=3)

   try:
       text = r.recognize_google(audio)
       return text

   except sr.UnknownValueError:
       return "Google Speech Recognition could not understand audio"

   except sr.RequestError as e:
       return "Could not request results from Google Speech Recognition service; {0}".format(e)


whiteboard = np.zeros((2000, 2000), np.uint8)
text=["Say Something"]
whiteboard.fill(255)
cv2.putText(whiteboard, "", (10, 100), cv2.FONT_HERSHEY_PLAIN, 4, 0, 3)
cv2.imshow("Board", whiteboard)
check=False
check_done=False
y=100
while(True):
   whiteboard.fill(255)
   y=100
   for t in text:
       cv2.putText(whiteboard, t, (10, y), cv2.FONT_HERSHEY_PLAIN, 4, 0, 3)
       y+=60
   cv2.imshow("Board", whiteboard)
   cv2.moveWindow("Board", 0, 700)

   key = cv2.waitKey(1)
   if(key==27):
       break
   r = sr.Recognizer()
   with sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size) as source:
       text.append(recog(source))
   if(not check):
       text=['Start Speaking']
       check=True
   elif(not check_done):
       text=text[1:]
       check_done=True
   key = cv2.waitKey(1)
   if(key==27):
       break



# while True:

cv2.destroyAllWindows()
