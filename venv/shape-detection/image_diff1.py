#!/bin/python3
# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy
from os import listdir
from os.path import isfile, join


def main(args):
    imageA = cv2.imread(args["image1"])
    imageA = cv2.resize(imageA, (50, 50))
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    (thresh, im_bwA) = cv2.threshold(grayA, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    mypath = 'Media/plates/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = numpy.empty(len(onlyfiles), dtype=object)
    result = []
    result_score = 0
    result_img = ""
    result_im_bw = ''
    for n in range(0, len(onlyfiles)):
        img_name = join(mypath, onlyfiles[n])
        imageB = cv2.imread(join(mypath, onlyfiles[n]))
        imageB = cv2.resize(imageB, (50, 50))
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (thresh, im_bwB) = cv2.threshold(grayB, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        (score, diff) = compare_ssim(im_bwA, im_bwB, full=True)
        #(score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        print("SSIM: {0}, image: {1}".format(score, img_name))

        result.append([score, img_name])
        if result_score < score:
            result_score = score
            result_img = img_name
            result_im_bw = im_bwB

    print (result_score, result_img)

    # show the output images
    cv2.imshow("Original", im_bwA)
    cv2.imshow("Match", result_im_bw)
    cv2.waitKey(0)



if __name__ == "__main__":
    main({"image1": "Media/well_0.png", "image2": "Media/Template/a_94.png"})
