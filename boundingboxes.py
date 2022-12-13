
import cv2
import numpy as np
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blob = cv2.dnn.blobFromImage(gray, 1 / 255, (416, 416), swapRB=True, crop=False)
model.setInput(blob)
outputs = model.forward()
bboxes = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            bbox = detection[0:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (x, y, w, h) = bbox.astype('int')
            bboxes.append([x, y, w, h])
print(bboxes)