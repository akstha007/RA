import cv2

stitcher = cv2.createStitcher(False)
foo = cv2.imread("Output/00001.png")
bar = cv2.imread("Output/00002.png")
result = stitcher.stitch((foo,bar))

cv2.imwrite("result.jpg", result[1])