import cv2
import numpy as np

image = cv2.imread("./img_data/mountain/match2.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# FAST Algo for corner detection
# Faster than SURF, good for real time detection. 
# However FAST gives us only the key points and we may need to compute descriptors with other algorithms like SIFT and SURF.
# With a Fast algorithm, we can detect corners and also blobs.
# But it is not robust to high levels of noise. It is dependent on a threshold.
# Syntax:
#           fast = cv.FastFeatureDetector_create()
#           fast.setNonmaxSuppression(False)
#           kp = fast.detect(gray_img, None)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=25)

# find and draw the keypoints
kp = fast.detect(image_gray,None)
img2 = cv2.drawKeypoints(image_gray, kp, None,color=(255,0,0))

print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imshow('fast_true.png',img2)
cv2.waitKey()

# Detecting multiple interest points in adjacent locations is another problem. It is solved by using Non-maximum Suppression
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(image_gray,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(image_gray, kp, None, color=(255,0,0))

cv2.imshow('fast_false.png',img3)
cv2.waitKey()

cv2.destroyAllWindows()