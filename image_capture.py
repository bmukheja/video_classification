import cv2

def capture_images_from_video():
    vidcap = cv2.VideoCapture('SampleVideo.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("captures/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: %d' %count, success)
        count += 1
    return


capture_images_from_video()