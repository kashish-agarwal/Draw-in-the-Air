# checks if the lower and upper bounds work
#kashish_agarwal
import cv2
import numpy as np
import time
w = 100
h = 200
lower_bound = np.array([0,80,50])
upper_bound = np.array([20,120,120])
values = np.array([])

cap = cv2.VideoCapture(0) # It converts video from the front camera into grayscale and display it
def centroid(cnt):
    #In OpenCV, moments are the average of the intensities of an image’s pixels.
    #OpenCV moments are used to describe several properties of an image, such as the intensity of an image, its centroid, the area, and information about its orientation.
    M = cv2.moments(cnt)
    cx = int(M['m10']/(M['m00']+1))
    cy = int(M['m01']/(M['m00']+1))
    return (cx, cy)

drawpts = []
while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    kernel = np.ones((7, 7), np.uint8)
    frame = cv2.erode(frame,kernel,iterations = 2)
    #erode() method is used to perform erosion on the image. The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of foreground object
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2. cvtColor() method is used to convert an image from one color space to another.
    # HSV is a cylindrical color model that remaps the RGB primary colors into dimensions that are easier for humans to understand. ... Hue specifies the angle of the color on the RGB color circle. A 0° hue results in red, 120° results in green, and 240° results in blue. Saturation controls the amount of color used.
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
    # Using inRange() in OpenCV to detect colors in a range
    res = cv2.bitwise_and(frame, frame, mask = mask)
    # AND of 2 images
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_contours=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>800:
            res=cv2.drawContours(res,[cnt],-1,(0,225,0),3)
            best_contours.append(cnt)

    if len(best_contours)>0:
        cent = []
        for cnt in best_contours:
            cent.append(centroid(cnt))
        mini=min(cent,key = lambda x: x[1])
        print(mini)
        drawpts.append(mini)
    else:
        print("no contours")

    for pt in drawpts:
        cv2.circle(frame,pt,3,(0,255,0),1)

    cv2.imshow('res', res)
    cv2.imshow('drawing',orig)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
