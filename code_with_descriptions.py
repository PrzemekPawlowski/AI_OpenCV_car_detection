import cv2

# Our video
video = cv2.VideoCapture('traffic.mp4')

# Our pre-trained car classifier
classifier_file = 'car_detector.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # Convert to grayscale (needed for haar cascade)
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect car
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the car
    # last parameter -> 2 this is a thickness of the rectangle
    for (x, y, width, height) in cars:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the image with cars
    cv2.imshow('Cars', frame)
    # Do not autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

