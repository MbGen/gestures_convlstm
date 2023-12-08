import os
import cv2
from time import sleep
from tqdm import tqdm

folder_index = 55
dirname = str(folder_index) + '/'

canny_def = 'canny_filter_default-30'
canny_swipe_up = 'canny_filter_swipe-up-30'

path_to_save = f'C:/Users/Andrew/Desktop/pythonProject/SwipeDetector/datasets/{canny_def}/{dirname}'

if os.path.exists(path_to_save):
    raise Exception("DIR IS ALREADY EXISTS")
else:
    os.mkdir(path_to_save)


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

counter = 0

# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'H264'), 10, (100, 200))

frames_counter = 0
frames_to_stop = 30

last_30_frames = []


def save_frames_to_folder(path, frames):
    for ix, fr in enumerate(frames):
        cv2.imwrite(path + f"{ix}.jpg", fr)


print(f'start recording')
sleep(1)

progress_bar = tqdm(total=frames_to_stop, desc='Processing frames', position=0)

record = False

while True:
    ret, frame = cap.read()
    reflected_image = cv2.flip(frame, 1)
    frame = reflected_image[50:250, 200:300]
    canny_frame = cv2.Canny(frame, 80, 150)

    # out.write(canny_frame)

    if ret:
        cv2.imshow('canny', canny_frame)

    if cv2.waitKey(5) == 32:
        record = True

    if record:
        cv2.waitKey(50)
        last_30_frames.append(canny_frame)
        progress_bar.update(1)

        if len(last_30_frames) == frames_to_stop:
            save_frames_to_folder(path_to_save, last_30_frames)
            record = False
            break

    elif cv2.waitKey(1) == 27:
        break

print('stop recording')

cv2.destroyWindow('canny')

# out.release()
