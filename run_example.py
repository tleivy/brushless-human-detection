from PersonDetector import *

detector = PersonDetector("model/keras_model.h5", "model/labels.txt")

while True:
    detector.detect("your_label")  # todo: use TL or omri for the example
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # key = Escape
        break

detector.kill_footage()
