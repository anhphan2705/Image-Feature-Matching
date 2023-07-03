import cv2

image = cv2.imread("./img_data/mountain/match2.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ORB (Oriented FAST and Rotated Brief)
# ORB is a very effective way to detect features compare to SIFT or SURF
# It detects fewer features but more important and in less time
# ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. 
# First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them. 
# It also use pyramid to produce multiscale-features.
# ORB use BRIEF descriptors. But we have already seen that BRIEF performs poorly with rotation. So what ORB does is to "steer" BRIEF according to the orientation of keypoints.

# Initiate ORB detector
orb = cv2.ORB_create()
# # find the keypoints with ORB
# kp = orb.detect(image_gray,None)
# # compute the descriptors with ORB
# kp, des = orb.compute(image_gray, kp)
kp, des = orb.detectAndCompute(image_gray, None) # or do both
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(image_gray, kp, None, color=(0,255,0), flags=0)

cv2.imshow("orb", img2)
cv2.waitKey()