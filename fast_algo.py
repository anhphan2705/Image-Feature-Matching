import cv2
import numpy as np

image = cv2.imread("./img_data/mountain/match2.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)