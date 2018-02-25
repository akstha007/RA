import time
import cv2
import os

def vid_2_frame(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    #video_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) - 1
    #print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    count = 0;
    num = 5; #skip no. of frames
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if (count % num == 0):
            if ret == True:
                # Write the results back to output location.
                cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
                count = count + 1
            else:
                # Log the time again
                time_end = time.time()
                # Release the feed
                cap.release()
                # Print stats
                print ("Done extracting frames.\n%d frames extracted" % count)
                print ("It took %d seconds forconversion." % (time_end-time_start))
                break

        count += 1

vid_2_frame("demo_vid.mp4","Output")