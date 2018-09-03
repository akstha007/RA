import time
import cv2
import os
import shutil

def convert(input_loc, output_loc="Output5", frame_rate=50):

    try:
        if os.path.isdir(output_loc):
            shutil.rmtree(output_loc)

        os.mkdir(output_loc)

        print("Directory created!!!")

    except OSError as exc:  # Guard against race condition
        print ("Error! Unable to create directory." + exc.message)

    time_start = time.time()
    cap = cv2.VideoCapture(input_loc)

    print ("Converting video...")
    count = 0;

    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if (count % frame_rate == 0):
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
