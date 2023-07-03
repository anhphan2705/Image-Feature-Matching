import cv2
import numpy as np

image = cv2.imread("./img_data/mountain/match1.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)                         # Converting the image to 8/W

# Harrish corner detection
# Taking vertical and horizontal derivative and look for where they are both high
# It detects the corner of an image by sliding a box all over a grayscale image. The corner will make marked on the image
# Syntax: cv2.cornerHarris(image, dest, blockSize, kSize, freeParameter, borderType)
#                          Image – The source image to detect the features
#                           Dest – Variable to store the output image
#                           Block size – Neighborhood size
#                           Ksize – Aperture parameter          # Increase the size of the Sobel's kernel
#                           freeParameter (k) - this is a predetermined value from 0.04 to 0.06 to filter out fake corners. Higher k = less corner, might filter out couple true corner. Lower k = more corner, miht also more fake ones.
#                           Border type: The pixel revealing type.

corners = cv2.cornerHarris(image_gray, blockSize=2, ksize=3, k=0.04)

# dilate to mark the corners
result_corners = cv2.dilate(corners, None)                   # Increase object area
image[result_corners > 0.01 * result_corners.max()] = [0, 255, 0]

cv2.imshow('haris_corner', image)
cv2.waitKey()




