import cv2
import numpy as np
import speech_recognition as sr
mic_name = "HDA Intel PCH: ALC3227 Analog (hw:0,0)"
sample_rate = 44100
chunk_size = 512
mic_list = sr.Microphone.list_microphone_names()
print(mic_list)
device_id=1
for i, microphone_name in enumerate(mic_list):
    if microphone_name == mic_name:
        device_id = i

print('DEVICE >> ')
print(device_id)

def recog(source):
    r.adjust_for_ambient_noise(source)
    print("Say Something...")
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)

        print("you said: " + text)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")

    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    r = sr.Recognizer()
    with sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size) as source:
        whiteboard = np.zeros((1000, 1000), np.uint8)
        whiteboard.fill(255)
        while True:
            cv2.imshow("Whiteboard", whiteboard)
            cv2.moveWindow("Whiteboard", 0, 0)
            recog(source)