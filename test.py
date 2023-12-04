import cv2
import numpy as np

# Загрузите изображение
image = cv2.imread('/home/humboy/PycharmProjects/pythonProject/SwipeDetector/datasets/swipe-up/1/1700238700.jpg',
                   cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(image, 50, 150)  # Здесь 50 и 150 - пороговые значения

# Отобразите исходное и обработанное изображение
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
