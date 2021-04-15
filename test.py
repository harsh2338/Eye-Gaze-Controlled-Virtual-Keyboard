import cv2
import numpy as np
import dlib
import math
import constants
from point import Point
import autocomplete
from direction import Direction
from keyboard_type import KeyboardType
from playsound import playsound
import time


class Eye():
    def __init__(self):

        self.init_detection()
        self.init_boards()
        self.gaze_direction = None
        self.keyboard_contents = None
        self.predicted_words = []
        self.text = ""
        self.prev_word = None
        self.algo()

    def init_detection(self):
        self.capture = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def get_updated_face(self):
        _, self.frame = self.capture.read()
        self.gray_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.detector(self.gray_img)

    def init_boards(self):
        self.keyboard = np.zeros((600, 1000, 3), np.uint8)
        self.keyboard.fill(255)

        self.whiteboard = np.zeros((300, 1000), np.uint8)
        self.whiteboard.fill(255)
        self.autocomplete_window = np.zeros((1000, 400), np.uint8)
        self.autocomplete_window.fill(255)
        self.gaze_direction = None
        self.keyboard_contents = None

    def get_eye_dimensions(self, extremes, top, bottom, landmarks):

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

        return eye_length, eye_height

    def is_blinking(self, landmarks):
        return self.is_left_wink(landmarks) and self.is_right_wink(landmarks)

    def is_left_wink(self, landmarks):
        left_len, left_ht = self.get_eye_dimensions(constants.LEFT_EYE_HORIZONTAL_EXTREMES, constants.LEFT_EYE_TOP,
                                                    constants.LEFT_EYE_BOTTOM, landmarks)
        return (left_len / left_ht > 4)

    def is_right_wink(self, landmarks):
        right_len, right_ht = self.get_eye_dimensions(constants.RIGHT_EYE_HORIZONTAL_EXTREMES, constants.RIGHT_EYE_TOP,
                                                      constants.RIGHT_EYE_BOTTOM, landmarks)
        return (right_len / right_ht > 4)

    def get_mid_point(self, p1, p2):
        return Point(int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2))

    def get_distance(self, p1, p2):
        return math.sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))

    def extract_eye_for_wink(self, eye_region):
        height, width, _ = self.frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, (0, 0, 255), 2)
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
        return white_count, threshold_eye

    def extract_eye(self, eye_region):
        height, width, _ = self.frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, (0, 0, 255), 2)
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

        wb_ratio = left_white / right_white
        return wb_ratio, threshold_eye

    def get_winked_eye_info(self, landmarks):
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

        left_white_count, left_thresh_eye = self.extract_eye_for_wink(left_eye_region)
        right_white_count, right_thresh_eye = self.extract_eye_for_wink(right_eye_region)

        # cv2.imshow(Direction.LEFT, left_thresh_eye)
        # cv2.imshow(Direction.RIGHT, right_thresh_eye)

        # print("\t\t\t\t\t\t\t\t\t\t",left_white_count,right_white_count,right_white_count/left_white_count)
        if (left_white_count < 10 and right_white_count > 10):
            cv2.putText(self.frame, 'Right Wink', (20, 180), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=3,
                        fontScale=1)
            return Direction.RIGHT
        elif (left_white_count > 10 and right_white_count < 10):

            cv2.putText(self.frame, 'Left Blink', (20, 180), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=3,
                        fontScale=1)
            return Direction.LEFT

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

        left_ratio, _ = self.extract_eye(left_eye_region)
        right_ratio, _ = self.extract_eye(right_eye_region)

        if (left_ratio < 1 and right_ratio < 1):
            cv2.putText(self.frame, "Right", (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=3,
                        fontScale=1)
            return Direction.RIGHT
        elif (left_ratio > 1.5 and right_ratio > 1.5):
            cv2.putText(self.frame, "Left", (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=3,
                        fontScale=1)
            return Direction.LEFT

        else:
            cv2.putText(self.frame, "Centre", (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0), thickness=3,
                        fontScale=1)
            return Direction.CENTRE

    def show_keyboard_contents(self, text, x, y, is_highlighted):
        width = 200
        height = 150
        th = 3

        if (is_highlighted):
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
        else:
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 0, 0), th)

        font_letter = cv2.FONT_HERSHEY_PLAIN
        font_scale = 10
        font_th = 4
        if (len(text) != 1):
            font_scale = 4
            font_th = 2
        text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
        width_text, height_text = text_size[0], text_size[1]
        text_x = int((width - width_text) / 2) + x
        text_y = int((height + height_text) / 2) + y
        cv2.putText(self.keyboard, text, (text_x, text_y), font_letter, font_scale, (0, 0, 0), font_th)

    def show_autocomplete_contents(self, text, x, y, is_highlighted):
        height = 50
        width = 400
        th = 3

        if (is_highlighted):
            cv2.rectangle(self.autocomplete_window, (x + 20, y + 20), (x + width - 20, y + height - 20), (0, 100, 100),
                          20)
        else:
            cv2.rectangle(self.autocomplete_window, (x + th, y + th), (x + width - th, y + height - th), (0, 255, 0),
                          th)

        font_letter = cv2.FONT_HERSHEY_PLAIN
        font_scale = 3
        font_th = 2

        text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
        width_text, height_text = text_size[0], text_size[1]
        text_x = int((width - width_text) / 2) + x
        text_y = int((height + height_text) / 2) + y
        cv2.putText(self.autocomplete_window, text, (text_x, text_y), font_letter, font_scale, (0, 0, 0), font_th)

    def draw_keyboard_window(self, highlight_index):
        index = 0
        for i in range(0, 600, 150):
            for j in range(0, 1000, 200):
                self.show_keyboard_contents(self.keyboard_contents[index], j, i, highlight_index == index)
                index += 1

    def draw_autocomplete_window(self, highlight_index):
        self.autocomplete_window.fill(255)

        index = 0
        if (len(self.predicted_words) > 0):

            for i in range(0, 1000, 100):
                self.show_autocomplete_contents(self.predicted_words[index], 0, i, highlight_index == index)
                index += 1
                if (index >= len(self.predicted_words)):
                    break

    def num_letter(self):
        rows, cols, _ = self.keyboard.shape
        th_lines = 4
        cv2.line(self.keyboard, (int(cols / 2) - int(th_lines / 2), 0), (int(cols / 2) - int(th_lines / 2), rows),
                 (0, 0, 0), th_lines)
        cv2.putText(self.keyboard, "Numeric", (80, 300), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0), thickness=3,
                    fontScale=3)
        cv2.putText(self.keyboard, "Letters", (80 + int(cols / 2), 300), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                    thickness=3,
                    fontScale=3)

    def show_options(self):
        rows, cols, _ = self.keyboard.shape
        th_lines = 4
        cv2.line(self.keyboard, (int(cols / 2) - int(th_lines / 2), 0), (int(cols / 2) - int(th_lines / 2), rows),
                 (0, 0, 0), th_lines)
        cv2.putText(self.keyboard, "ENIHR", (80, 200), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0), thickness=3,
                    fontScale=3)
        cv2.putText(self.keyboard, "UCW", (120, 300), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0), thickness=3,
                    fontScale=3)
        cv2.putText(self.keyboard, "YBVJX", (80, 400), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0), thickness=3,
                    fontScale=3)

        cv2.putText(self.keyboard, "TAOSD", (80 + int(cols / 2), 200), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                    thickness=3,
                    fontScale=3)
        cv2.putText(self.keyboard, "LMF", (140 + int(cols / 2), 300), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                    thickness=3,
                    fontScale=3)

        cv2.putText(self.keyboard, "GPKQZ", (80 + int(cols / 2), 400), cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                    thickness=3,
                    fontScale=3)

    def update_cursor(self, frame_counter, highlight_index, cursor_direction):
        if (frame_counter % constants.FPS == 0):
            if (cursor_direction == Direction.LEFT_TO_RIGHT):
                highlight_index += 1
            else:
                highlight_index -= 1
            highlight_index = highlight_index % len(self.keyboard_contents)
            frame_counter = 0

        return frame_counter, highlight_index

    def generate_autocomplete_words(self):
        autocomplete.load()
        words = self.text.split(' ')
        print(words)
        self.predicted_words = []
        if (len(words) > 1):

            while (words[-1] == ''):
                words.pop(-1)

            self.prev_word = words[-2]
            cur_word = words[-1]
            self.predicted_words = autocomplete.predict(self.prev_word.lower(), cur_word.lower())
        else:
            cur_word = words[-1]
            self.predicted_words = autocomplete.predict(cur_word.lower(), '')

        self.predicted_words = [i.upper() for i, j in self.predicted_words]
        self.predicted_words = self.predicted_words[:9]
        self.predicted_words.insert(0, '<-')

    def algo(self):
        highlight_index = 0

        blinking_counter = 0
        gaze_counter = 0
        wink_counter = 0
        frame_counter = 0

        autocomplete_counter = 0
        autocomplete_cursor_index = 0

        cursor_direction = Direction.LEFT_TO_RIGHT

        is_keyboard_selected = False
        is_in_autocomplete_window = False
        is_winking = True

        prev_gaze = None
        prev_wink = None

        while True:
            self.get_updated_face()
            self.keyboard.fill(255)
            self.whiteboard.fill(255)
            #
            # if (len(self.text) > 0):
            #     if (self.text[0] == ""):
            #         text_to_display = self.text[1:]
            #     else:
            #         text_to_display = self.text
            #     cv2.putText(self.whiteboard, text_to_display, (10, 100), cv2.FONT_HERSHEY_PLAIN, 4, 0, 3)
            #
            # self.show_options()
            #
            # if (len(self.faces) < 1):
            #     continue
            #
            # face = self.faces[0]
            # landmarks = self.predictor(self.gray_img, face)
            # self.gaze_direction = self.get_gaze_direction(landmarks)
            # self.draw_autocomplete_window(-1)
            #
            # if (is_in_autocomplete_window):
            #     self.keyboard.fill(255)
            #     autocomplete_counter += 1
            #
            #     if (autocomplete_counter % constants.FPS == 0):
            #         autocomplete_cursor_index += 1
            #
            #         autocomplete_cursor_index = autocomplete_cursor_index % len(self.predicted_words)
            #         autocomplete_counter = 0
            #         self.draw_autocomplete_window(autocomplete_cursor_index)
            #
            #     self.draw_keyboard_window(highlight_index)
            #
            #     # ---
            #     # key = cv2.waitKey(1)
            #     # if key == 32:
            #     # ---
            #
            #     # ************
            #     # if (self.is_blinking(landmarks)):
            #     #     if(blinking_counter==constants.FPS):
            #     #         words=self.text.split(' ')[:-1]
            #     #         self.text=" ".join(words)+" "+self.predicted_words[autocomplete_cursor_index]+ " "
            #     #         print(self.text)
            #     #         frame_counter -= 1
            #     #         blinking_counter = 0
            #     #         wink_counter = 0
            #     #         is_in_autocomplete_window = False
            #     #         autocomplete_counter = 0
            #     #         autocomplete_cursor_index = -1
            #     # *************
            #     # else:
            #     #     cv2.putText(self.frame, 'Eye closed', (20, 250), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0), thickness=3,
            #     #                 fontScale=1)
            #     # blinking_counter+=1
            #
            #     # else:
            #     #     blinking_counter=0
            #
            #     eye_which_winked = self.get_winked_eye_info(landmarks)
            #
            #     if (eye_which_winked != None):
            #         if (prev_wink == None):
            #             prev_wink = eye_which_winked
            #
            #         # if (eye_which_winked == Direction.LEFT):
            #         #     if (prev_wink == eye_which_winked):
            #         #         wink_counter += 1
            #         #     else:
            #         #         wink_counter = 0
            #         #
            #         #     if (wink_counter > constants.FPS):
            #         #         cv2.putText(self.frame, 'Left Wink', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
            #         #                     thickness=3,
            #         #                     fontScale=1)
            #         #         wink_counter = 0
            #         #
            #         if (eye_which_winked == Direction.RIGHT):
            #             cv2.putText(self.frame, 'Right Wink', (20, 150), cv2.FONT_HERSHEY_COMPLEX,
            #                         color=(255, 0, 0),
            #                         thickness=3,
            #                         fontScale=1)
            #
            #             if (prev_wink == eye_which_winked):
            #                 wink_counter += 1
            #             else:
            #                 wink_counter = 0
            #
            #             if (wink_counter > constants.FPS):
            #                 playsound('wink.wav')
            #                 if (self.predicted_words[autocomplete_cursor_index] != '<-'):
            #                     words = self.text.split(' ')[:-1]
            #                     self.text = " ".join(words) + " " + self.predicted_words[
            #                         autocomplete_cursor_index] + " "
            #                 blinking_counter = 0
            #                 wink_counter = 0
            #                 is_in_autocomplete_window = False
            #                 is_keyboard_selected = True
            #                 autocomplete_counter = 0
            #                 autocomplete_cursor_index = -1
            #                 prev_gaze = None
            #                 self.generate_autocomplete_words()
            #         prev_wink = eye_which_winked
            #
            #
            # elif (is_keyboard_selected):
            #
            #     frame_counter += 1
            #     frame_counter, highlight_index = self.update_cursor(frame_counter, highlight_index, cursor_direction)
            #     self.keyboard.fill(255)
            #     self.draw_keyboard_window(highlight_index)
            #
            #     if (prev_gaze == None):
            #         prev_gaze = self.gaze_direction
            #
            #     if (self.gaze_direction == Direction.LEFT):
            #         if (prev_gaze == self.gaze_direction):
            #             gaze_counter += 1
            #         else:
            #             gaze_counter = 0
            #
            #         if (gaze_counter > constants.FPS):
            #             cv2.putText(self.frame, "Left", (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
            #                         thickness=3,
            #                         fontScale=1)
            #             cursor_direction = Direction.RIGHT_TO_LEFT
            #             blinking_counter = 0
            #             gaze_counter = 0
            #
            #     elif (self.gaze_direction == Direction.RIGHT):
            #         if (prev_gaze == self.gaze_direction):
            #             gaze_counter += 1
            #         else:
            #             gaze_counter = 0
            #         if (gaze_counter > constants.FPS):
            #             cv2.putText(self.frame, "Right", (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0),
            #                         thickness=3,
            #                         fontScale=1)
            #             cursor_direction = Direction.LEFT_TO_RIGHT
            #             blinking_counter = 0
            #             gaze_counter = 0
            #
            #     prev_gaze = self.gaze_direction
            #
            #     if (self.is_blinking(landmarks)):
            #         # time.sleep(2)
            #         if (blinking_counter == constants.FPS):
            #             playsound('sound.wav')
            #             cv2.putText(self.frame, 'Selected', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
            #                         thickness=3,
            #                         fontScale=1)
            #             if (self.keyboard_contents[highlight_index] == '123'):
            #                 self.gaze_direction = KeyboardType.NUMERIC
            #                 self.keyboard_contents = constants.NUMBERS
            #                 frame_counter = 1
            #             elif (self.keyboard_contents[highlight_index] == 'Right'):
            #                 self.gaze_direction = Direction.RIGHT
            #                 self.keyboard_contents = constants.RIGHT_LETTERS
            #                 frame_counter = 1
            #             elif (self.keyboard_contents[highlight_index] == 'Left'):
            #                 self.gaze_direction = Direction.LEFT
            #                 self.keyboard_contents = constants.LEFT_LETTERS
            #                 frame_counter = 1
            #             elif (self.keyboard_contents[highlight_index] == 'Auto'):
            #                 if (len(self.text) > 0 and len(self.predicted_words) > 0):
            #                     is_in_autocomplete_window = True
            #             elif (self.keyboard_contents[highlight_index] == 'Space'):
            #                 self.text += " "
            #             else:
            #                 self.text += self.keyboard_contents[highlight_index]
            #                 frame_counter = 1
            #             self.generate_autocomplete_words()
            #             highlight_index = 0
            #             cursor_direction = Direction.LEFT_TO_RIGHT
            #             time.sleep(1)
            #         else:
            #             cv2.putText(self.frame, 'Eye closed', (20, 250), cv2.FONT_HERSHEY_COMPLEX, color=(255, 0, 0),
            #                         thickness=3,
            #                         fontScale=1)
            #         blinking_counter += 1
            #         frame_counter -= 1
            #
            #     else:
            #         blinking_counter = 0
            #
            #     self.get_eye_dimensions(constants.LEFT_EYE_HORIZONTAL_EXTREMES, constants.LEFT_EYE_TOP,
            #                             constants.LEFT_EYE_BOTTOM, landmarks)
            #     self.get_eye_dimensions(constants.RIGHT_EYE_HORIZONTAL_EXTREMES, constants.RIGHT_EYE_TOP,
            #                             constants.RIGHT_EYE_BOTTOM, landmarks)
            #
            #     eye_which_winked = self.get_winked_eye_info(landmarks)
            #
            #     if (eye_which_winked != None):
            #
            #         if (prev_wink == None):
            #             prev_wink = eye_which_winked
            #
            #         if (eye_which_winked == Direction.LEFT):
            #             #
            #             if (prev_wink == eye_which_winked):
            #                 wink_counter += 1
            #             else:
            #                 wink_counter = 0
            #
            #             if (wink_counter > constants.FPS):
            #                 cv2.putText(self.frame, 'Left Wink', (20, 150), cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0),
            #                             thickness=3,
            #                             fontScale=1)
            #                 wink_counter = 0
            #
            #         elif (eye_which_winked == Direction.RIGHT):
            #             # playsound('wink.wav')
            #             if (prev_wink == eye_which_winked):
            #                 wink_counter += 1
            #             else:
            #                 wink_counter = 0
            #
            #             if (wink_counter > constants.FPS):
            #                 cv2.putText(self.frame, 'Right Wink', (20, 150), cv2.FONT_HERSHEY_COMPLEX,
            #                             color=(255, 0, 0),
            #                             thickness=3,
            #                             fontScale=1)
            #                 blinking_counter = 0
            #                 wink_counter = 0
            #                 if (len(self.text) > 0 and len(self.predicted_words) > 0):
            #                     is_in_autocomplete_window = True
            #         prev_wink = eye_which_winked
            #
            # else:
            #     self.show_options()
            #
            #     if (prev_gaze == None):
            #         prev_gaze = self.gaze_direction
            #
            #     if (self.gaze_direction == Direction.LEFT):
            #         if (prev_gaze == self.gaze_direction):
            #             gaze_counter += 1
            #         else:
            #             gaze_counter = 0
            #         if (gaze_counter > constants.FPS):
            #             self.keyboard_contents = constants.LEFT_LETTERS
            #             self.draw_keyboard_window(highlight_index)
            #             playsound('sound.wav')
            #
            #             is_keyboard_selected = True
            #             highlight_index = 0
            #             gaze_counter = 0
            #
            #     elif (self.gaze_direction == Direction.RIGHT):
            #         if (prev_gaze == self.gaze_direction):
            #             gaze_counter += 1
            #         else:
            #             gaze_counter = 0
            #         if (gaze_counter > constants.FPS):
            #             playsound('sound.wav')
            #             self.keyboard_contents = constants.RIGHT_LETTERS
            #             self.draw_keyboard_window(highlight_index)
            #             is_keyboard_selected = True
            #             highlight_index = 0
            #             gaze_counter = 0
            #
            #     prev_gaze = self.gaze_direction
            #
            #     self.get_eye_dimensions(constants.LEFT_EYE_HORIZONTAL_EXTREMES, constants.LEFT_EYE_TOP,
            #                             constants.LEFT_EYE_BOTTOM, landmarks)
            #     self.get_eye_dimensions(constants.RIGHT_EYE_HORIZONTAL_EXTREMES, constants.RIGHT_EYE_TOP,
            #                             constants.RIGHT_EYE_BOTTOM, landmarks)
            self.keyboard_contents = constants.LEFT_LETTERS
            self.draw_keyboard_window(0)
            cv2.imshow("Frame", self.frame)
            cv2.moveWindow("Frame", 0, 0)
            cv2.imshow("Keyboard", self.keyboard)
            cv2.moveWindow("Keyboard", 500, 0)
            cv2.imshow("Board", self.whiteboard)
            cv2.moveWindow("Board", 500, 1000)
            cv2.imshow("Autocomplete Window", self.autocomplete_window)
            cv2.moveWindow("Autocomplete Window", 1510, 0)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif (key == 32):
                is_in_autocomplete_window = True

        self.capture.release()
        cv2.destroyAllWindows()


def main():
    autocomplete.load()
    Eye()


main()


