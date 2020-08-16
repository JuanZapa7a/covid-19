# import the necessary packages
import cv2

class AddChannelPreprocessor:
    def __init__(self, minVal=60, maxVal=120):
        # minval and maxval are Canny arguments
        self.minVal = minVal
        self.maxVal = maxVal

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        R = image
        image = cv2.medianBlur(R,5)
        G = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
        B = cv2.Canny(image, self.minVal, self.maxVal)

        return cv2.merge([B, G, R])
