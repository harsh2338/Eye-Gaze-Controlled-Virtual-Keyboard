import cv2
import numpy as np
import speech_recognition as sr
#
# class SpeechRecognition():

mic_name = "HDA Intel PCH: ALC3227 Analog (hw:0,0)"
sample_rate = 44100
chunk_size = 512
mic_list = sr.Microphone.list_microphone_names()
device_id=1
for i, microphone_name in enumerate(mic_list):
    if microphone_name == mic_name:
        device_id = i

def recog(source):
    r.adjust_for_ambient_noise(source)
    print("Say Something...")
    audio = r.listen(source,timeout=1,phrase_time_limit=1)

    try:
        text = r.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"

    except sr.RequestError as e:
        return "Could not request results from Google Speech Recognition service; {0}".format(e)


whiteboard = np.zeros((2000, 2000), np.uint8)
text="Say Something"
while(True):
    whiteboard.fill(255)
    cv2.putText(whiteboard, text, (10, 100), cv2.FONT_HERSHEY_PLAIN, 4, 0, 3)
    cv2.imshow("Board", whiteboard)
    # cv2.moveWindow("Board", 500, 1000)
    r = sr.Recognizer()
    # source=sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size)
    # recog(source)
    with sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size) as source:
        text=(recog(source))

    key = cv2.waitKey(1)
    if(key==27):
        break



# while True:

cv2.destroyAllWindows()
