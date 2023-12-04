import os

import cv2
import time

folder_index = 50
dirname = str(folder_index) + '/'

# canny_def = 'canny_filter_default'
# canny_swipe_up = 'canny_filter_swipe-up'
#
# path_to_save = f'C:/Users/Andrew/Desktop/pythonProject/SwipeDetector/datasets/{canny_def}/{dirname}'
#
# if os.path.exists(path_to_save):
#     raise Exception("DIR IS ALREADY EXISTS")
# else:
#     os.mkdir(path_to_save)


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

counter = 0

# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'H264'), 10, (100, 200))

while True:
    ret, frame = cap.read()
    reflected_image = cv2.flip(frame, 1)
    frame = reflected_image[50:250, 200:300]
    canny_frame = cv2.Canny(frame, 80, 150)

    # out.write(canny_frame)

    if ret:
        cv2.imshow('canny', canny_frame)

    # if cv2.waitKey(1) == 32:
        # filename = f'{time.time():.0f}.jpg'
        # result = cv2.imwrite(path_to_save + filename, canny_frame)
        # if result:
        #     counter += 1
        # print(f'space was clicked {counter} times')

    elif cv2.waitKey(1) == 27:
        break

cv2.destroyWindow('canny')

# out.release()
