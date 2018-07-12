import sys
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    # Get width and height of input images	
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation 
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))

    # result_img[transform_dist[1]:w1 + transform_dist[1],
    # transform_dist[0]:h1 + transform_dist[0]] = img1

    x_range = range(transform_dist[1], (w1 + transform_dist[1]))
    y_range = range(transform_dist[0], (h1 + transform_dist[0]))

    print(result_img.shape)

    for i in x_range:
        for j in y_range:
            # if np.all(result_img[i,j,:])!=0:
            if result_img[i, j, 0] != 0 and result_img[i, j, 1] != 0 and result_img[i, j, 2] != 0:
                result_img[i][j] = img1[i - transform_dist[1]][j - transform_dist[0]]

    # Return the result
    return result_img


# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
    # Initialize SIFT 
    sift = cv2.AKAZE_create()
    # sift = cv2.xfeatures2d.SIFT_create()

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    # Make sure that the matches are good
    verify_ratio = 0.8  # Source: stackoverflow
    verified_matches = []
    for m1, m2 in matches:
        # Add to array only if it's a good match
        if m1.distance < verify_ratio * m2.distance:
            verified_matches.append(m1)

    print(len(verified_matches))

    # Mimnum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 3.0)
        # M, mask = cv2.findHomography(img1_pts, img2_pts)

        # M = cv2.getAffineTransform(img1_pts, img2_pts)
        # M = [[1,2,3],[1,2,3]]
        # M = np.vstack((M,[0,0,1]))
        # print("this is M value:", M)

        return M
    else:
        print 'Error: Not enough matches'
        exit()


# Equalize Histogram of Color Images
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


def combine_image():
    img1 = cv2.imread("Output4/00001.jpg");
    img2 = cv2.imread("Output4/00051.jpg");

    res = np.concatenate((img1, img2), axis=1)
    cv2.imwrite("result.jpg", res)
    # cv2.imshow("Result", res);
    cv2.waitKey(0)


def combine_images():
    mypath = 'Output3/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)
    res = cv2.imread("Output3/00001.jpg")
    # res = cv2.resize(res, (320, 240))
    for n in range(1, len(onlyfiles)):
        img2 = cv2.imread(join(mypath, onlyfiles[n]))
        # img2 = cv2.resize(img2, (320, 240))

        res = np.concatenate((res, img2), axis=1)

    result_image_name = "result.jpg"
    cv2.imwrite(result_image_name, res)

    # Show the resulting image
    print("Image combined!!!")
    # cv2.imshow("Result", res);
    # cv2.waitKey(0)


def paranoma():
    mypath = 'Output3/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)
    # img1 = cv2.imread("Output3/00001.jpg")
    # img1 = cv2.resize(img1, (640, 480))
    for n in range(0, len(onlyfiles) - 1):
        print(n, join(mypath, onlyfiles[n]))

        img1 = cv2.imread(join(mypath, onlyfiles[n]))
        img1 = cv2.resize(img1, (640, 480))

        img2 = cv2.imread(join(mypath, onlyfiles[n + 1]))
        img2 = cv2.resize(img2, (640, 480))

        # Use SIFT to find keypoints and return homography matrix
        M = get_sift_homography(img1, img2)

        # Stitch the images together using homography matrix
        result_img = get_stitched_image(img2, img1, M)

        result_image_name = "result" + str(n) + ".jpg"
        cv2.imwrite(result_image_name, result_img)

    # Show the resulting image
    print("Image converted!!!")


def paranoma1():
    mypath = 'Output4/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)
    img1 = cv2.imread("Output4/00001.jpg")
    img1 = cv2.resize(img1, (320, 240))
    for n in range(1, 6):
        print(n, join(mypath, onlyfiles[n]))

        img2 = cv2.imread(join(mypath, onlyfiles[n]))
        img2 = cv2.resize(img2, (320, 240))

        # img1 = equalize_histogram_color(img1)
        # img2 = equalize_histogram_color(img2)

        # Use SIFT to find keypoints and return homography matrix
        M = get_sift_homography(img1, img2)
        print(M)

        # Stitch the images together using homography matrix
        img1 = get_stitched_image(img2, img1, M)

    result_image_name = "result.jpg"
    cv2.imwrite(result_image_name, img1)

    # Show the resulting image
    print("Image converted!!!")


def paranoma2():
    mypath = 'Output4/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)
    dirname = create_output_dir()

    res_img = cv2.imread("Output4/00001.jpg")
    res_img = image_resize(res_img, height=240)

    M_list = []

    for n in range(0, 8):  # len(onlyfiles) - 1):
        print(n, join(mypath, onlyfiles[n]))

        img1 = cv2.imread(join(mypath, onlyfiles[n]))
        img1 = image_resize(img1, height=240)

        img2 = cv2.imread(join(mypath, onlyfiles[n + 1]))
        img2 = image_resize(img2, height=240)

        # img1 = equalize_histogram_color(img1)
        # img2 = equalize_histogram_color(img2)

        # Use SIFT to find keypoints and return homography matrix
        M = get_sift_homography(img1, img2)
        M = np.linalg.inv(M)
        print("Image1 Image2: ", M)

        if n != 0:
            mult = np.dot(M_list[-1], M)
            print("Mult of all M's: ", mult)
        else:
            mult = M

        M_list.append(mult)

        M_res = get_sift_homography(res_img, img2)
        print("Result Image2: ", M_res)

        # Stitch the images together using homography matrix
        result = get_stitched_image(img1, img2, M)
        result_image_name = "result" + str(n) + ".jpg"
        cv2.imwrite(os.path.join(dirname, result_image_name), result)

        # reverse M_res
        # M_res = np.linalg.inv(M_res)

        res_img = get_stitched_image(img2, res_img, mult)
        result_image_name = "Res_mix" + str(n) + ".jpg"
        cv2.imwrite(os.path.join(dirname, result_image_name), res_img)

    # Show the resulting image
    print("Image converted!!!")


def create_output_dir():
    dirname = 'Output_imgs'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def paranoma3():
    mypath = 'Output4/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)
    dirname = create_output_dir()

    res_img = cv2.imread("Output4/00001.jpg")
    res_img = image_resize(res_img, height=240)

    M_list = []

    for n in range(0, 8):  # len(onlyfiles) - 1):
        print(n, join(mypath, onlyfiles[n]))

        img1 = cv2.imread(join(mypath, onlyfiles[n]))
        img1 = image_resize(img1, height=240)

        img2 = cv2.imread(join(mypath, onlyfiles[n + 1]))
        img2 = image_resize(img2, height=240)

        # img1 = equalize_histogram_color(img1)
        # img2 = equalize_histogram_color(img2)

        # Use SIFT to find keypoints and return homography matrix
        M = get_sift_homography(img1, img2)
        M = np.linalg.inv(M)
        print("Image1 Image2: ", M)

        if n != 0:
            mult = np.dot(M_list[-1], M)
            print("Mult of all M's: ", mult)
        else:
            mult = M

        M_list.append(mult)

        M_res = get_sift_homography(res_img, img2)
        print("Result Image2: ", M_res)

        # Stitch the images together using homography matrix
        result = get_stitched_image(img1, img2, M)
        result_image_name = "result" + str(n) + ".jpg"
        cv2.imwrite(os.path.join(dirname, result_image_name), result)

        # reverse M_res
        # M_res = np.linalg.inv(M_res)

        res_img = get_stitched_image(img2, res_img, mult)
        result_image_name = "Res_mix" + str(n) + ".jpg"
        cv2.imwrite(os.path.join(dirname, result_image_name), res_img)

    # Show the resulting image
    print("Image converted!!!")


# Main function definition
def main():
    paranoma3()
    # combine_images()


# Call main function
if __name__ == '__main__':
    main()
