
import cv2
# our image 
img_file = 'c2.jpg'
video = cv2.VideoCapture('street.mp4')

#our pre-trained car classifier
car_tracker_file = 'car_detector.xml'

#pedestrian tracker 
pedestrian_tracker_file ='pedestrain.xml'

#create the car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        #must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #car  and pedestrian detector 
    cars = car_tracker.detectMultiScale(grayscaled_frame) 
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

     #draw rectangles around the cars
    for (x, y, w, h)in cars:
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 100), 2)

     #draw rectangles around the pedestrian
    for (x, y, w, h)in pedestrians:
       cv2.rectangle(frame, (x, y), (x+w, y+h), ( 255, 255, 255), 2)


  #display the iamge with the faces spotted
    cv2.imshow('car detector',frame)

    #dont autoclose 
    cv2.waitKey(1)
