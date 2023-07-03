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
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(image_gray, None)
  

# Applying the function
kp_image = cv2.drawKeypoints(image, kp, None, color=(
    0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT', kp_image)
cv2.waitKey()

# SURF (Speeded-Up Robust Features)
# SURF is faster than SIFT
# SURF goes a little further than SIFT and approximates LoG with Box Filter
# One big advantage of this approximation is that, convolution with box filter can be easily calculated with the help of integral images. 
# And it can be done in parallel for different scales. 
# Also the SURF rely on determinant of Hessian matrix for both scale and location
# It is 3 times faster than SIFT while performance is comparable to SIFT. 
# SURF is good at handling images with blurring and rotation, but not good at handling viewpoint change and illumination change
# Syntax:
#       surf = cv2.SURF_create()
#       kp, des = sift.detectAndCompute(image_gray, None)

### Patented cant be use ###

# surf = cv2.xfeatures2d.SURF_create()
# kp, des = sift.detectAndCompute(image_gray, None)

# kp_image = cv2.drawKeypoints(image, kp, None, color=(
#     0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow('SURF', kp_image)
# cv2.waitKey()
