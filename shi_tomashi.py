import cv2
import numpy as np

image = cv2.imread("./img_data/mountain/match2.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Shi-Tomashi corner detection
# Almost the same as Harris corner detection
# The only difference is in the kernel value in which we can only find n strongest corner of the image
# This is very crucial when we need a limited but important features of the image.
# Syntax: cv2.goodFeatureToTrack(image, maxc, Quality, maxD)
#           Imagev - the source image 
#           maxc - maximun corners that we want, negative value gives all the corners
#           Quality - quality level parameter (preferred value = 0.01)
#           minD - minimun distance (preferred value = 10)

corners = cv2.goodFeaturesToTrack(image_gray, maxCorners=50, qualityLevel=0.02, minDistance=20)
corners = np.float32(corners)

# Circle and mark the corners green
for item in corners:
    x, y = item[0]
    x = int(x)
    y = int(y)
    cv2.circle(image, (x,y), 6, (0, 255, 0), -1)
    
cv2.imshow("shi_tomashi", image)
cv2.waitKey()