
import cv2
# our image 
img_file = 'c2.jpg'

#our pre-trained car classifier
classifier_file = 'car_detector.xml'

#create opencv image
img = cv2.imread(img_file)


#convert to the grayscale(needed for haar cascader)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#CREATE Classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#DETECT cars
cars = car_tracker.detectMultiScale(black_n_white)

#draw rectangles around the cars
for (x, y, w, h)in cars:
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 55), 2)



#display the iamge with the faces spotted
cv2.imshow('car detector',img)

#dont autoclose 
cv2.waitKey()

print("code complited")
