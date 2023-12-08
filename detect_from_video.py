import torch
import torch.nn as nn
import cv2

cap = cv2.VideoCapture('output2.mp4')


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

        nn.init.kaiming_normal_(self.fc1.weight)

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)

    def forward(self, x):
        B = x.shape[0]

        x = self.maxpool(self.relu(self.conv1(x)))
        x = nn.functional.dropout(x, p=0.3, training=self.training)
        x = self.maxpool(self.relu(self.conv2(x)))
        x = nn.functional.dropout(x, p=0.3, training=self.training)
        x = x.view(B, -1)

        x = self.tanh(self.fc(x))

        h_t, c_t = self.lstm(x)
        h_t = nn.functional.dropout(h_t, p=0.3, training=self.training)
        x = self.tanh(self.fc1(h_t))

        return self.softmax(x), h_t, c_t


model = ConvLSTM(num_classes_to_predict=2)
model.load_state_dict(torch.load('canny_v4_drop', map_location=torch.device('cpu')))
model.eval()

last_5_frames = []
last_5_frames_numpy = []


def show_prediction(frames, label, probs=None):
    if probs is not None:
        print(label, probs)
    else:
        print(label)

    combined_images = cv2.hconcat(frames)
    cv2.imshow(label, combined_images)
    cv2.waitKey(2000)
    cv2.destroyWindow(label)

frames_to_skip = 10
frames_to_skip_counter = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    rgb_to_single_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_for_torch = (torch.tensor(rgb_to_single_channel) / 255).view(1, 200, 100)
    frames_to_skip_counter += 1

    if frames_to_skip_counter < frames_to_skip:
        continue

    last_5_frames.append(frame_for_torch)
    last_5_frames_numpy.append(rgb_to_single_channel)
    frames_to_skip_counter = 0

    if len(last_5_frames) == 5:
        inp = torch.stack(last_5_frames, dim=0)
        inp = inp.view(1, 1, 5, 200, 100)

        predict, _, _ = model(inp)

        prediction_label = 'DEFAULT' if predict.argmax(dim=1) == torch.tensor([0]) else 'SWIPE-UP'
        show_prediction(last_5_frames_numpy, prediction_label, predict)

    elif len(last_5_frames) > 5:
        last_5_frames.clear()
        last_5_frames_numpy.clear()

    cv2.imshow('Video Playback', rgb_to_single_channel)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
