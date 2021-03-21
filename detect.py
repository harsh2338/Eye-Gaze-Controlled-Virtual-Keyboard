import cv2
import numpy as np
import dlib
import math
import constants
from point import Point
import autocomplete
import time

class Eye():
    def __init__(self):

        self.capture = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        _, self.frame = self.capture.read()
        self.gray_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.detector(self.gray_img)
        self.keyboard = np.zeros((600, 1000, 3), np.uint8)
        self.keyboard.fill(255)
        self.board = np.zeros((300, 1000), np.uint8)
        self.board.fill(255)
        self.current_selected_input_type=None
        self.keyboard_contents=None
        self.counter=0
        self.text=""
        self.direction="L2R"

        self.algo()
    def get_eye_dimensions(self,extremes,top,bottom,landmarks):

        left_point = Point(landmarks.part(extremes[0]).x,
                           landmarks.part(extremes[0]).y)
        right_point = Point(landmarks.part(extremes[1]).x,
                            landmarks.part(extremes[1]).y)

        center_top = self.get_mid_point(landmarks.part(top[0]), landmarks.part(top[1]))
        center_bottom = self.get_mid_point(landmarks.part(bottom[0]),
                                      landmarks.part(bottom[1]))

        cv2.line(self.frame, (left_point.x, left_point.y), (right_point.x, right_point.y), (0, 0, 255), 2)
        cv2.line(self.frame, (center_top.x, center_top.y), (center_bottom.x, center_bottom.y), (0, 0, 255), 2)

        eye_height = self.get_distance(center_bottom, center_top)
        eye_length = self.get_distance(left_point, right_point)

        return eye_length,eye_height
    def is_blinking(self,landmarks):
        return self.is_left_wink(landmarks) and self.is_right_wink(landmarks)

    def is_left_wink(self,landmarks):
        left_len,left_ht=self.get_eye_dimensions(constants.LEFT_EYE_HORIZONTAL_EXTREMES,constants.LEFT_EYE_TOP,constants.LEFT_EYE_BOTTOM,landmarks)
        return (left_len/left_ht>4)

    def is_right_wink(self,landmarks):
        right_len,right_ht=self.get_eye_dimensions(constants.RIGHT_EYE_HORIZONTAL_EXTREMES,constants.RIGHT_EYE_TOP,constants.RIGHT_EYE_BOTTOM,landmarks)
        return (right_len/right_ht>4)

    def get_mid_point(self,p1,p2):
        return Point(int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))
    def get_distance(self,p1,p2):
        return math.sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2))

    def extract_eye_for_wink(self,eye_region):
        height, width, _ = self.frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, (0,0,255), 2)
        cv2.fillPoly(mask, [eye_region], 255)
        left_eye = cv2.bitwise_and(self.gray_img, self.gray_img, mask=mask)

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])
        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 25, 255, cv2.THRESH_BINARY)

        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        # eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        # threshold_eye_row_num, threshold_eye_col_num = threshold_eye.shape
        # left_half = threshold_eye[0:threshold_eye_row_num, 0:int(threshold_eye_col_num / 2)]
        # right_half = threshold_eye[0:threshold_eye_row_num, int(threshold_eye_col_num / 2):threshold_eye_col_num]

        white_count = max(1, cv2.countNonZero(threshold_eye))

        # wb_ratio=left_white / right_white
        return  white_count,threshold_eye

    def extract_eye(self,eye_region):
        height, width, _ = self.frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, (0,0,255), 2)
        cv2.fillPoly(mask, [eye_region], 255)
        left_eye = cv2.bitwise_and(self.gray_img, self.gray_img, mask=mask)

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])
        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY)

        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        threshold_eye_row_num, threshold_eye_col_num = threshold_eye.shape
        left_half = threshold_eye[0:threshold_eye_row_num, 0:int(threshold_eye_col_num / 2)]
        right_half = threshold_eye[0:threshold_eye_row_num, int(threshold_eye_col_num / 2):threshold_eye_col_num]

        left_white = max(1, cv2.countNonZero(left_half))
        right_white = max(1, cv2.countNonZero(right_half))

        wb_ratio=left_white / right_white
        return  wb_ratio,threshold_eye

    def get_winked_eye_info(self,landmarks):
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        left_white_count,left_thresh_eye= self.extract_eye_for_wink(left_eye_region)
        right_white_count,right_thresh_eye= self.extract_eye_for_wink(right_eye_region)


        # cv2.imshow("Left", left_thresh_eye)
        # cv2.imshow("Right", right_thresh_eye)

        # print("\t\t\t\t\t\t\t\t\t\t",left_white_count,right_white_count,right_white_count/left_white_count)
        if(left_white_count<10 and right_white_count>10):
            cv2.putText(self.frame, 'Right Wink', (20, 180), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=3,
                        fontScale=1)
            return "Right"
        elif(left_white_count>10 and right_white_count<10):

            cv2.putText(self.frame, 'Left Blink', (20, 180), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=3,
                    fontScale=1)
            return "Left"

        return None


    def get_gaze_direction(self, landmarks):
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        left_ratio,_= self.extract_eye(left_eye_region)
        right_ratio,_= self.extract_eye(right_eye_region)




        if(left_ratio<1 and right_ratio<1):
            cv2.putText(self.frame, 'Right', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=3,
                        fontScale=1)
            return "Right"
        elif(left_ratio>1.5 and right_ratio>1.5):
            cv2.putText(self.frame, 'Left', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=3,
                        fontScale=1)
            return "Left"

        else:
            cv2.putText(self.frame, 'Centre', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0), thickness=3,
                        fontScale=1)
            return "Centre"


    def show_contents(self,text, x, y,is_highlighted):
        width = height = 200
        th = 3

        if(is_highlighted):
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
        else:
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 0, 0), th)

        font_letter = cv2.FONT_HERSHEY_PLAIN
        font_scale = 10
        font_th = 4
        if(len(text)!=1):
            font_scale=4
            font_th=2
        text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
        width_text, height_text = text_size[0], text_size[1]
        text_x = int((width - width_text) / 2) + x
        text_y = int((height + height_text) / 2) + y
        cv2.putText(self.keyboard, text, (text_x, text_y), font_letter, font_scale, (0, 0, 0), font_th)

    def draw_window(self,highlight_index):
        index=0
        for i in range(0, 600, 200):
            for j in range(0, 1000, 200):
                self.show_contents(self.keyboard_contents[index], j, i, highlight_index == index)
                index += 1

    def num_letter(self):
        rows, cols, _ = self.keyboard.shape
        th_lines = 4
        cv2.line(self.keyboard, (int(cols / 2) - int(th_lines / 2), 0), (int(cols / 2) - int(th_lines / 2), rows),
                 (0,0,0), th_lines)
        cv2.putText(self.keyboard, "Numeric", (80, 300), cv2.FONT_HERSHEY_COMPLEX,color=(0, 0, 0),thickness=3,
                                        fontScale=3)
        cv2.putText(self.keyboard, "Letters", (80 + int(cols / 2), 300), cv2.FONT_HERSHEY_COMPLEX,color=(0, 0, 0),thickness=3,
                                        fontScale=3)
    def show_options(self):
        rows, cols, _ = self.keyboard.shape
        th_lines = 4
        cv2.line(self.keyboard, (int(cols / 2) - int(th_lines / 2), 0), (int(cols / 2) - int(th_lines / 2), rows),
                 (0,0,0), th_lines)
        cv2.putText(self.keyboard, "ENIHR", (80, 200), cv2.FONT_HERSHEY_COMPLEX,color=(0, 0, 0),thickness=3,
                                        fontScale=3)
        cv2.putText(self.keyboard, "UCW", (120, 300), cv2.FONT_HERSHEY_COMPLEX,color=(0, 0, 0),thickness=3,
                                        fontScale=3)
        cv2.putText(self.keyboard, "YBVJX", (80, 400), cv2.FONT_HERSHEY_COMPLEX,color=(0, 0, 0),thickness=3,
                                        fontScale=3)

        cv2.putText(self.keyboard, "TAOSD", (80 + int(cols / 2), 200), cv2.FONT_HERSHEY_COMPLEX,color=(0, 0, 0),thickness=3,
                                        fontScale=3)
        cv2.putText(self.keyboard, "LMF", (140 + int(cols / 2), 300), cv2.FONT_HERSHEY_COMPLEX,color=(0, 0, 0),thickness=3,
                                        fontScale=3)

        cv2.putText(self.keyboard, "GPKQZ", (80 + int(cols / 2), 400), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                    thickness=3,
                    fontScale=3)
    def algo(self):
        highlight_index=0
        blinking_counter=0
        gaze_counter=0
        is_keyboard_selected=False
        prev_gaze=None

        while True:
            _, self.frame = self.capture.read()
            rows, cols, _ = self.frame.shape

            self.keyboard.fill(255)
            self.gray_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # self.frame[rows - 50: rows, 0: cols] = (255, 255, 255)
            self.faces = self.detector(self.gray_img)
            self.show_options()

            if(is_keyboard_selected):

                self.counter+=1
                if(self.counter%constants.FPS==0):
                    if(self.direction=='L2R'):
                        highlight_index+=1
                    else:
                        highlight_index-=1
                    highlight_index=highlight_index%len(self.keyboard_contents)
                    self.counter=0

                self.keyboard.fill(255)

                for face in self.faces:
                    landmarks = self.predictor(self.gray_img, face)

                    self.draw_window(highlight_index)

                    self.current_selected_input_type = self.get_gaze_direction(landmarks)

                    if(prev_gaze==None):
                        prev_gaze=self.current_selected_input_type

                    if (self.current_selected_input_type == 'Left'):
                        if(prev_gaze==self.current_selected_input_type):
                            gaze_counter+=1
                        else:
                            gaze_counter=0
                        # gaze_counter+=1
                        if(gaze_counter>constants.FPS):
                            cv2.putText(self.frame, 'Left', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
                                        thickness=3,
                                        fontScale=1)
                            self.direction="R2L"
                            blinking_counter = 0
                            gaze_counter=0

                    elif (self.current_selected_input_type == 'Right'):
                        if(prev_gaze==self.current_selected_input_type):
                            gaze_counter+=1
                        else:
                            gaze_counter=0
                        # gaze_counter += 1
                        if (gaze_counter > constants.FPS):
                            cv2.putText(self.frame, 'Right', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0),
                                        thickness=3,
                                        fontScale=1)
                            self.direction="L2R"
                            blinking_counter = 0
                            gaze_counter=0

                    prev_gaze=self.current_selected_input_type

                    eye_which_winked=self.get_winked_eye_info(landmarks)
                    if(eye_which_winked=='Left'):
                        self.current_selected_input_type = 'Left'
                        self.keyboard_contents = constants.RIGHT_LETTERS
                    elif(eye_which_winked=='Right'):
                        self.current_selected_input_type = 'Right'
                        self.keyboard_contents = constants.LEFT_LETTERS


                    if (self.is_blinking(landmarks)):

                        if(blinking_counter==constants.FPS):
                            cv2.putText(self.frame, 'Selected', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
                                        thickness=3,
                                        fontScale=1)
                            if(self.keyboard_contents[highlight_index]=='123'):
                                self.current_selected_input_type='Numeric'
                                self.keyboard_contents=constants.NUMBERS
                                self.counter = 1
                            elif(self.keyboard_contents[highlight_index]=='Right'):
                                self.current_selected_input_type = 'Right'
                                self.keyboard_contents = constants.RIGHT_LETTERS
                                self.counter = 1
                                # self.text+='?'
                            elif(self.keyboard_contents[highlight_index]=='Left'):
                                self.current_selected_input_type = 'Left'
                                self.keyboard_contents = constants.LEFT_LETTERS
                                self.counter = 1
                                # is_keyboard_selected=False
                            else:
                                self.text+=self.keyboard_contents[highlight_index]
                                print(self.text,autocomplete.predict('',str(self.text)))
                                self.counter = 1
                            highlight_index = 0
                            self.direction="L2R"
                        else:
                            cv2.putText(self.frame, 'Eye closed', (20, 250), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0), thickness=3,
                                        fontScale=1)
                        blinking_counter+=1
                        self.counter-=1
                    else:
                        blinking_counter=0

                    self.get_eye_dimensions(constants.LEFT_EYE_HORIZONTAL_EXTREMES, constants.LEFT_EYE_TOP,
                                            constants.LEFT_EYE_BOTTOM, landmarks)
                    self.get_eye_dimensions(constants.RIGHT_EYE_HORIZONTAL_EXTREMES, constants.RIGHT_EYE_TOP,
                                            constants.RIGHT_EYE_BOTTOM, landmarks)
                cv2.putText(self.board, self.text, (10, 100), cv2.FONT_HERSHEY_PLAIN, 4, 0, 3)

            else:
                for face in self.faces:
                    landmarks = self.predictor(self.gray_img, face)
                    self.show_options()
                    self.current_selected_input_type = self.get_gaze_direction(landmarks)

                    if(prev_gaze==None):
                        prev_gaze=self.current_selected_input_type

                    if (self.current_selected_input_type == 'Left'):
                        if(prev_gaze==self.current_selected_input_type):
                            gaze_counter+=1
                        else:
                            gaze_counter=0
                        if(gaze_counter>constants.FPS):
                            self.keyboard_contents=constants.LEFT_LETTERS
                            self.draw_window(highlight_index)
                            is_keyboard_selected=True
                            highlight_index = 0
                            blinking_counter = 0
                            gaze_counter=0
                            prev_gaze=None

                    elif (self.current_selected_input_type == 'Right'):
                        if(prev_gaze==self.current_selected_input_type):
                            gaze_counter+=1
                        else:
                            gaze_counter=0
                        if (gaze_counter > constants.FPS):
                            self.keyboard_contents=constants.RIGHT_LETTERS
                            self.draw_window(highlight_index)
                            is_keyboard_selected=True
                            highlight_index = 0
                            blinking_counter = 0
                            gaze_counter=0
                            prev_gaze=None

                    prev_gaze=self.current_selected_input_type

                    self.get_eye_dimensions(constants.LEFT_EYE_HORIZONTAL_EXTREMES, constants.LEFT_EYE_TOP,
                                    constants.LEFT_EYE_BOTTOM, landmarks)
                    self.get_eye_dimensions(constants.RIGHT_EYE_HORIZONTAL_EXTREMES, constants.RIGHT_EYE_TOP,
                                    constants.RIGHT_EYE_BOTTOM, landmarks)
            cv2.imshow("Frame", self.frame)
            cv2.moveWindow("Frame", 0, 0)
            cv2.imshow("Keyboard", self.keyboard)
            cv2.moveWindow("Keyboard", 500, 0)
            cv2.imshow("Board", self.board)
            cv2.moveWindow("Board", 500, 1000)
            key = cv2.waitKey(1)
            if key == 27:
                break
        self.capture.release()
        cv2.destroyAllWindows()

def main():
    # sound = pyglet.media.load("click.wav", streaming=False)
    autocomplete.load()
    Eye()


main()

