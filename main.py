import cv2

video = cv2.VideoCapture('traffic.mp4')

classifier_file = 'car_detector.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, width, height) in cars:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv2.imshow('Cars', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
