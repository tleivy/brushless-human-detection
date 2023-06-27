import cv2
import numpy as np
from keras.models import load_model


class PersonDetector:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path)  # TODO: create model
        self.labels = open(labels_path, 'r').readlines()
        self.camera = cv2.VideoCapture(0)

    def detect(self, desired_label):
        """
        Wrap it in a while loop.
        example:
        while True:
            person_detector.detect()
        :param desired_label: String = 'Person'
        :return: True if detected a Person
        """

        ret, image = self.camera.read()

        if not ret:
            raise IOError

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # --- DEBUG FOOTAGE WINDOW ---
        cv2.imshow('Webcam Image', image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Have the model predict what the current image is.
        # returns an array of percentages.
        # Example:[0.2,0.8] meaning its 20% sure it is the first label and 80% sure it's the second label.
        probabilities = self.model.predict(image)

        # Get the label of the highest probability detection
        detected = (self.labels[np.argmax(probabilities)])

        return detected.strip() == desired_label.strip()
    

    def kill_footage(self):
        self.camera.release()
        cv2.destroyAllWindows()  # --- KILL DEBUG FOOTAGE WINDOW ---

