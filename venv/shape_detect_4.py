#!/bin/python3
import numpy as np
import cv2
import imutils
from pyimagesearch.shapedetector import ShapeDetector
import numpy
from os import listdir, mkdir, rmdir
from os.path import isfile, join, isdir
from skimage.measure import compare_ssim
from scipy import ndimage
import errno
import vid2frame
import shutil

# inputs
frame_rate = 50
vid_loc = "Media/well_vid2.mp4"
frame_dir = "test/"  # this dir contains frames of the video
template_img = "raw/"  # this dir must be present containing well plate no. samples

# panaroma images folder
panaroma_dir = "panaroma/"

well_w = 210  # 210
well_h = 210
tmp_img = "tmp.jpg"
result_img = "well_plate.jpg"
tmp_well_dir = "tmp_well_dir/"
tmp_out_dir = "tmp_out_dir/"

# shape of well plate numbering
well_plate = [["a", "y", "z", "d", "e", "f"],
              ["g", "h", "i", "j", "k", "l"],
              ["m", "n", "o", "p", "q", "r"],
              ["s", "t", "u", "v", "w", "x"]]

result_list = {}
result_list_files = {}


def create_dirs(tmp_well_dir=tmp_well_dir, tmp_out_dir=tmp_out_dir):
    try:
        if isdir(tmp_well_dir):
            shutil.rmtree(tmp_well_dir)

        mkdir(tmp_well_dir)

        if isdir(tmp_out_dir):
            shutil.rmtree(tmp_out_dir)

        mkdir(tmp_out_dir)

        if isdir(panaroma_dir):
            shutil.rmtree(panaroma_dir)

        mkdir(panaroma_dir)

        print("Directory created!!!")

    except OSError as exc:  # Guard against race condition
        print ("Error! Unable to create directory." + exc.message)


def match_shape(imgA):
    imageA = cv2.imread(imgA)
    imageA = cv2.resize(imageA, (50, 50))
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    (thresh, im_bwA) = cv2.threshold(grayA, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image, contours, hierarchy = cv2.findContours(im_bwA, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mypath = template_img
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    result_score = 0
    result_img = ""
    for n in range(0, len(onlyfiles)):
        img_name = join(mypath, onlyfiles[n])
        imageB = cv2.imread(join(mypath, onlyfiles[n]))

        imageB = cv2.resize(imageB, (50, 50))
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (thresh, im_bwB) = cv2.threshold(grayB, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        image, contours, hierarchy = cv2.findContours(im_bwB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        (score, diff) = compare_ssim(im_bwA, im_bwB, full=True)

        if score > result_score:
            result_score = score
            result_img = img_name

    return result_img, result_score


def image_rotate(img, angle):
    return ndimage.rotate(img, angle)


def detect_well_final(image_name, tmp_img=tmp_img, thres_low=100):
    orig_image = cv2.imread(image_name)
    if orig_image.shape[0] > orig_image.shape[1]:
        resized = imutils.resize(orig_image, width=240)
    else:
        resized = imutils.resize(orig_image, height=240)
        resized = image_rotate(resized, -90)

    imgray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(imgray, thres_low, 200, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image.shape

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000 and area < 3000:
            [x, y, w, h] = cv2.boundingRect(c)
            if h in range(30, 50) and w in range(30, 50):
                # result contins valid image
                r_w = float(img_w) / (x + 25)
                r_h = float(img_h) / (y + 25)

                if r_w > 1.5 and r_w < 2.5 and r_h > 1.5 and r_h < 2.5:
                    # save the crop_image
                    crop_image = resized[y:y + w, x:x + w]
                    img_name = tmp_img

                    # compare with raw/images
                    cv2.imwrite(img_name, crop_image)
                    result, result_score = match_shape(img_name)

                    if len(result) > 1:
                        print("result-final: ", result_score, [x, y, w, h])
                        result_name = tmp_well_dir + result.split("/")[1]
                        x = x + 17
                        y = y + 17
                        well_wby2 = well_w // 2
                        well_hby2 = well_h // 2

                        if y > well_hby2 and y < img_h - well_hby2 and x > well_wby2 and x < img_w - well_wby2:
                            #result_list[result_name] = result_score
                            result_well = resized[y - well_hby2:y + well_hby2,
                                          x - well_wby2:x + well_wby2]
                            cv2.imwrite(result_name, result_well)
                            # store original image in panaroma dir
                            result_name = panaroma_dir + result.split("/")[1]
                            cv2.imwrite(result_name, orig_image)


def detect_well(image_name, tmp_img=tmp_img, thres_low=100):
    orig_image = cv2.imread(image_name)

    if orig_image.shape[0] > orig_image.shape[1]:
        resized = imutils.resize(orig_image, width=240)
    else:
        resized = imutils.resize(orig_image, height=240)
        resized = image_rotate(resized, -90)

    imgray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(imgray, thres_low, 200, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image.shape

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000 and area < 3000:
            [x, y, w, h] = cv2.boundingRect(c)
            if h in range(30, 50) and w in range(30, 50):
                # result contins valid image
                r_w = float(img_w) / (x + 25)
                r_h = float(img_h) / (y + 25)

                if r_w > 1.5 and r_w < 2.5 and r_h > 1.5 and r_h < 2.5:
                    # save the crop_image
                    crop_image = resized[y:y + w, x:x + w]
                    img_name = tmp_img

                    # compare with raw/images
                    cv2.imwrite(img_name, crop_image)
                    result, result_score = match_shape(img_name)

                    if len(result) > 1:
                        print("result: ", result_score, [x, y, w, h])
                        result_name = tmp_well_dir + result.split("/")[1]
                        x = x + 17
                        y = y + 17
                        well_wby2 = well_w // 2
                        well_hby2 = well_h // 2

                        # check if index of image present in list
                        if result_name in result_list:
                            # check if new image has more similarity then already detected image if present
                            if result_list[result_name] < result_score:
                                if y > well_hby2 and y < img_h - well_hby2 and x > well_wby2 and x < img_w - well_wby2:
                                    result_list[result_name] = result_score
                                    result_list_files[result_name] = image_name
                                    result_well = resized[y - well_hby2:y + well_hby2, x - well_wby2:x + well_wby2]
                                    cv2.imwrite(result_name, result_well)
                                    #store original image in panaroma dir
                                    result_name = panaroma_dir + result.split("/")[1]
                                    cv2.imwrite(result_name, orig_image)
                        else:
                            if y > well_hby2 and y < img_h - well_hby2 and x > well_wby2 and x < img_w - well_wby2:
                                result_list[result_name] = result_score
                                result_list_files[result_name] = image_name
                                result_well = resized[y - well_hby2:y + well_hby2, x - well_wby2:x + well_wby2]
                                cv2.imwrite(result_name, result_well)
                                # store original image in panaroma dir
                                result_name = panaroma_dir + result.split("/")[1]
                                cv2.imwrite(result_name, orig_image)


def combine_images(tmp_out_dir=tmp_out_dir, result_img=result_img):
    mypath = tmp_out_dir
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)
    res = cv2.imread(join(mypath, onlyfiles[0]))
    for n in range(1, len(onlyfiles)):
        img2 = cv2.imread(join(mypath, onlyfiles[n]))
        res = np.concatenate((res, img2), axis=0)

    cv2.imwrite(result_img, res)

    # Show the resulting image
    print("Well plate generated!!!")


def combine_images_row(width=well_w, height=well_h, tmp_out_dir=tmp_out_dir, tmp_well_dir=tmp_well_dir):
    mypath = tmp_well_dir
    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image[:, :] = (255, 255, 255)  # (B, G, R)
    # border of image
    blank_image[:, :1] = (0, 0, 0)
    blank_image[:, -1:] = (0, 0, 0)
    blank_image[:1, :] = (0, 0, 0)
    blank_image[-1:, :] = (0, 0, 0)

    for i in range(len(well_plate)):
        img_path = mypath + well_plate[i][0] + ".jpg"
        if isfile(img_path):
            res = cv2.imread(img_path)
        else:
            res = blank_image

        for j in range(1, len(well_plate[0])):
            img_path = mypath + well_plate[i][j] + ".jpg"
            if isfile(img_path):
                img2 = cv2.imread(img_path)
            else:
                img2 = blank_image

            res = np.concatenate((res, img2), axis=1)

        result_image_name = tmp_out_dir + str(i) + ".jpg"
        cv2.imwrite(result_image_name, res)

    # Show the resulting image
    print("Row image created!!!")


def generate_wells(frame_dir=frame_dir, thres_low=100):
    mypath = frame_dir
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for n in range(0, len(onlyfiles)):
        imageA = join(mypath, onlyfiles[n])
        detect_well(imageA, thres_low=thres_low)


def generate_wells_final(thres_low=100):
    result_list = {}
    for key, value in result_list_files.items():
        detect_well_final(value, thres_low=thres_low)


def main():
    create_dirs(tmp_well_dir=tmp_well_dir, tmp_out_dir=tmp_out_dir)
    # vid2frame.convert(input_loc=vid_loc, output_loc=frame_dir, frame_rate=frame_rate)

    for i in range(80, 150, 10):
        generate_wells(frame_dir=frame_dir, thres_low=i)

    #for i in range(80, 150, 10):
    #    generate_wells_final(thres_low=i)

    combine_images_row(width=well_w, height=well_h, tmp_out_dir=tmp_out_dir, tmp_well_dir=tmp_well_dir)
    combine_images(tmp_out_dir=tmp_out_dir, result_img=result_img)


if __name__ == "__main__":
    main()
