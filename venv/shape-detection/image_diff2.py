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
    mypath = args
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    sim_arrs = []
    # diff_arrs = []
    for i in range(0, len(onlyfiles)):
        sim_arr = []
        # diff_arr = []
        for j in range(0, len(onlyfiles)):
            imageA = cv2.imread(join(mypath, onlyfiles[i]))
            imageA = cv2.resize(imageA, (50, 50))
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            (thresh, im_bwA) = cv2.threshold(grayA, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            imageB = cv2.imread(join(mypath, onlyfiles[j]))
            imageB = cv2.resize(imageB, (50, 50))
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            (thresh, im_bwB) = cv2.threshold(grayB, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            (score, diff) = compare_ssim(im_bwA, im_bwB, full=True)

            sim_arr.append("{0:.2f}".format(score))
            # diff_arr.append(diff)

        sim_arrs.append(sim_arr)
        # diff_arrs.append(diff_arr)

    print("Similarity:\n")
    for row in sim_arrs:
        print(row)

    # print("\n---------------------------------------\n")
    # print("Disimilarity\n")
    # print(diff_arrs)


if __name__ == "__main__":
    main("Media/plates/")
