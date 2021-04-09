# {
#   "apikey": "7QJbUbg1kfUnUKz5Smc0MMJFJYe0EFYd8ftfT7oQxVT9",
#   "iam_apikey_description": "Auto-generated for key 0e55df32-743c-4b8c-8be2-1f1e93b940ec",
#   "iam_apikey_name": "Auto-generated service credentials",
#   "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
#   "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/fe233168381b4fc9ac7441485684a2b1::serviceid:ServiceId-70dbdc57-1e09-4764-bd76-d040bc82c8d0",
#   "url": "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/13252e47-6bb7-41b9-a247-8a88fb83b804"
# }
import cv2
import numpy as np
class Eye():
    def __init__(self):
        self.init_boards()
        self.algo()


    def init_boards(self):
        self.whiteboard = np.zeros((300, 1000), np.uint8)
        self.whiteboard.fill(255)

    def algo(self):
        while True:
            self.whiteboard.fill(255)
            cv2.imshow("Board", self.whiteboard)
            cv2.moveWindow("Board", 500, 1000)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cv2.destroyAllWindows()

def main():
    Eye()


main()


