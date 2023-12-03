import os
# region delete

import torch
import torch.nn as nn

# endregion

import cv2
import time

# dirname = '11' + '/'
#
# path_to_save = '/home/humboy/PycharmProjects/pythonProject/SwipeDetector/datasets/canny_filter_swipe-up/' + dirname
#
# if os.path.exists(path_to_save):
#     raise Exception("DIR IS ALREADY EXISTS")
# else:
#     os.mkdir(path_to_save)


# region should be deleted
class ConvLSTM(nn.Module):
    def __init__(self, num_classes_to_predict):
        super().__init__()

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1, 10, 10))

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 5, 5))

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(1, 5, 5))

        self.fc = nn.Linear(960, 2000)

        self.lstm = nn.LSTM(2000, 1000, batch_first=True)

        self.fc1 = nn.Linear(1000, num_classes_to_predict)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]

        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))

        x = x.view(B, -1)

        x = self.tanh(self.fc(x))

        h_t, c_t = self.lstm(x)

        x = self.tanh(self.fc1(h_t))

        return self.softmax(x), h_t, c_t

# endregion

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)


model = ConvLSTM(num_classes_to_predict=2)
model.load_state_dict(torch.load('canny_v3', map_location=torch.device('cpu')))
model.eval()
counter = 0

last_5_frames = []

while True:
    ret, frame = cap.read()
    reflected_image = cv2.flip(frame, 1)
    frame = reflected_image[50:250, 200:300]
    canny_frame = cv2.Canny(frame, 80, 150)

    frame_for_torch = (torch.tensor(canny_frame) / 255).view(1, 200, 100)

    last_5_frames.append(frame_for_torch)

    cv2.waitKey(200)

    if len(last_5_frames) == 5:
        inp = torch.stack(last_5_frames, dim=0)
        inp = inp.view(1, 1, 5, 200, 100)
        predict, _, _ = model(inp)
        print('DEFAULT' if predict.argmax(dim=1) == torch.tensor([0]) else 'SWIPE-UP', end='\n\n\n\n')
    elif len(last_5_frames) > 10:
        last_5_frames.clear()

    if ret:
        # cv2.imshow('Камера', frame)
        cv2.imshow('canny', canny_frame)

    # if cv2.waitKey(1) == 32:
    #     filename = f'{time.time():.0f}.jpg'
    #     result = cv2.imwrite(path_to_save + filename, canny_frame)
    #     if result:
    #         counter += 1
    #     print(f'space was clicked {counter} times')

    elif cv2.waitKey(1) == 27:
        break

cv2.destroyWindow('canny')
