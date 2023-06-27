from PersonDetector import *

detector = PersonDetector(r"model/keras_model.h5", r"model/labels.txt")

while True:
    detector.detect("your_label")  # for example: "person"

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # key = Escape
        break

detector.kill_footage()
