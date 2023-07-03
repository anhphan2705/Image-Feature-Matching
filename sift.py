import cv2
import numpy as np

image = cv2.imread("./img_data/mountain/match2.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT (Scale-Invariant Feature Transform)
# Algos that detects object irrelavant to the scale and rotation of the image and reference
# Return key points in the image to mark
# Syntax:   sift = cv2.xfeatures2D.SIFT_create()
#           key_point, descripter =  sift.detectAndCompute(gray_image, optionValue=None)

# Applying the function
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(image_gray, None)
  
  
# Applying the function
kp_image = cv2.drawKeypoints(image, kp, None, color=(
    0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT', kp_image)
cv2.waitKey()