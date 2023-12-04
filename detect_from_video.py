import torch
import torch.nn as nn
import cv2

cap = cv2.VideoCapture('output_video.mp4')

class ConvLSTM(nn.Module):
    pass


model = ConvLSTM(num_classes_to_predict=2)
model.load_state_dict(torch.load('canny_v3', map_location=torch.device('cpu')))
model.eval()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Video Playback', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
