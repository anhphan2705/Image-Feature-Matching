import cv2
import numpy as np

image = cv2.imread("./img_data/mountain/match1.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)                         # Converting the image to 8/W

# Harrish corner detection
# It detects the corner of an image by sliding a box all over. The corner will make marked on the image
# Syntax: cv2.cornerHarris(image, dest, blockSize, kSize, freeParameter, borderType)
#                          Image – The source image to detect the features
#                           Dest – Variable to store the output image
#                           Block size – Neighborhood size
#                           Ksize – Aperture parameter
#                           Border type: The pixel revealing type.

result_img = cv2.cornerHarris(image_gray, blockSize=2, ksize=3, k=0.04)

# dilate to mark the corners
result_img = cv2.dilate(result_img, None)
image[result_img > 0.01 * result_img.max()] = [0, 255, 0]

cv2.imshow('haris_corner', image)
cv2.waitKey()

