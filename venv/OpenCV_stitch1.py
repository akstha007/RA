import cv2

try:
    stitcher = cv2.createStitcher(False)
    foo = cv2.imread("Output/00001.jpg")
    bar = cv2.imread("Output/00266.jpg")

    #foo = cv2.imread("result.jpg")
    #bar = cv2.imread("result2.jpg")

    result = stitcher.stitch((foo,bar))
    cv2.imwrite("result2.jpg", result[1])

except Exception as e:
    print(e)