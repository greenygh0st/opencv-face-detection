import cv2, os
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

recognizer = cv2.createLBPHFaceRecognizer()
path = './training'

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.md') and not f.endswith('.DS_Store')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = face_cascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            print "Face found appending to set"
            #cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
            #cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            print "Face detected."

            # Identify the face here
            image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.md') and not f.endswith('.DS_Store')]
            #print len(image_paths)
            #if len(image_paths) == 0:
                #print 'Path empty ' + path
            for image_path in image_paths:
                #trying image
                print 'Trying image'
                predict_image_pil = Image.open(image_path).convert('L')
                predict_image = np.array(predict_image_pil, 'uint8')
                nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
                nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
                if nbr_actual == nbr_predicted:
                    print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
                else:
                    print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)

            #draw around the face
            cv2.rectangle(img, (x,y), (x+w, y+h), (2555,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


print 'Starting up'
print 'Building dataset'
# fetch the images
images, labels = get_images_and_labels(path)

# Perform the tranining
recognizer.train(images, np.array(labels))

#main()
