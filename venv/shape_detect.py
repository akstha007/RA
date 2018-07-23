#!/bin/python3
import numpy as np
import cv2
import imutils
from pyimagesearch.shapedetector import ShapeDetector
import numpy
from os import listdir
from os.path import isfile, join
from skimage.measure import compare_ssim


def match_shape(imgA):
    imageA = cv2.imread(imgA)
    #imageB = cv2.imread('raw/b.jpg')

    imageA = cv2.resize(imageA, (50, 50))
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    (thresh, im_bwA) = cv2.threshold(grayA, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image, contours, hierarchy = cv2.findContours(im_bwA, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = contours[0]

    mypath = 'raw/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = numpy.empty(len(onlyfiles), dtype=object)
    result = []
    result_score = 0
    result_img = ""
    for n in range(0, len(onlyfiles)):
        img_name = join(mypath, onlyfiles[n])
        imageB = cv2.imread(join(mypath, onlyfiles[n]))

        imageB = cv2.resize(imageB, (50, 50))
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (thresh, im_bwB) = cv2.threshold(grayB, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        image, contours, hierarchy = cv2.findContours(im_bwB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt2 = contours[0]

        #ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
        (score, diff) = compare_ssim(im_bwA, im_bwB, full=True)

        if score>result_score:
            result_score = score
            result_img = img_name

    print result_img, result_score
    return result_img

def main(args):
    image = cv2.imread(args["image"])
    if image.shape[0] > image.shape[1]:
        resized = imutils.resize(image, width=240)
    else:
        resized = imutils.resize(image, height=240)

    sd = ShapeDetector()
    imgray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 0;
    # loop over the contours
    for c in contours:
        area = cv2.contourArea(c)
        if area > 200:
            [x, y, w, h] = cv2.boundingRect(c)
            #cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
            if h in range(30, 50) and w in range(30,50):
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]) - w / 2.5)
                cY = int((M["m01"] / M["m00"]) + h / 1.5)
                shape = sd.detect(c)
                print(shape, cX, cY)

                if shape in ["square", "rectangle","pentagon"]:
                    # save the crop_image
                    crop_image = resized[y:y + w, x:x + w]

                    #img_name = "Media/well_{0}.png".format(count)
                    #count += 1
                    img_name = "test.jpg"

                    #compare with raw/images
                    cv2.imwrite(img_name, crop_image)
                    result = match_shape(img_name)

                    # draw contour
                    cv2.drawContours(resized, [c], -1, (255, 0, 0), 2)
                    cv2.putText(resized, result, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    #draw rectangle for well
                    cv2.rectangle(resized, (x-90, y-80), (x+125, y+135), (255, 0, 0), 2)

    cv2.imshow('final image', resized)
    cv2.waitKey(0)


if __name__ == "__main__":
    #main({"image": "Output4/00501.jpg"})
    main({"image": "00001.jpg"})

