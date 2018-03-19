#!/bin/python3
import numpy as np
import cv2
import imutils
from pyimagesearch.shapedetector import ShapeDetector


def main(args):
    image = cv2.imread(args["image"])
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    sd = ShapeDetector()

    imgray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # thresh = cv2.adaptiveThreshold(imgray, 255, 1, 1, 11, 2)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(resized, contours, -1, (0, 255, 0), 3)

    # loop over the contours
    for c in contours:
        area = cv2.contourArea(c)
        #print area
        if area > 500:
            [x, y, w, h] = cv2.boundingRect(c)
            if h > 30 and h < 80 and w > 35:
                #print("my area", area)
                #print(x, y, w, h)
                #cv2.drawContours(resized, [c], -1, (0, 255, 0), 3)
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                shape = sd.detect(c)
                print(shape)

                if(shape == "rectangle" or shape == "pentagon"):
                    cv2.drawContours(resized, [c], -1, (0, 255, 0), 3)
                    cv2.putText(resized, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)

                # show the output image
                #cv2.imshow("Image", resized)
                #cv2.waitKey(0)

    cv2.imshow('final image', resized)
    cv2.waitKey(0)


if __name__ == "__main__":
    # main({"image": "images/shapes_and_colors.png"})
    # main({"image": "test2-1-invert.jpg"})
    main({"image": "Media/img4.jpg"})
