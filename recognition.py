#region libraries
import cv2
import face_recognition
import numpy as np
import os
#endregion


#region variables

path = 'images'  # The name of folder of images
images = [] # an empty list for putting each known image in it 
names = [] # an empty list for putting each name of images
my_list = [] # an empty list for the path
my_list = os.listdir(path) # read the images in the folder
# print(my_list)

#endregion


#region known images
for item in my_list: # make a loop for read each known images 
    img_current = cv2.imread(f'{path}/{item}') # read the image by opencv
    images.append(img_current) # append each knwon images in images list
    names.append(item.split('.')[0]) # append each name of images in names list

# print(images)

#endregion


#region known encodes
def encoding(images): # Make a function for encode known images
    encode_list = [] # an empty list for put each knwon encode in it

    for image in images: # make a loop for take each image and encode it
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the color of images to encode them
        encode = face_recognition.face_encodings(image)[0] # encode the images
        encode_list.append(encode) # append each encode in the encode_list

    return encode_list 

known_encode = encoding(images) # put the function in another variable 

# print(len(known_encode)) # prent the lenth of images

#endregion


#region new faces
cap = cv2.VideoCapture(0) # for read frames from webcam

while (cap.isOpened()): # While the webcam works correctly and if has no problems for webcam , Then loop will work
    ret,frame = cap.read() # read each frame from webcam
    resize = cv2.resize(frame, (0,0), None , 0.25 , 0.25) # resize the frame
    color = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB) # convert the color of frame to RGB
    faces_loc = face_recognition.face_locations(color) # find the location of each face
    faces_encode = face_recognition.face_encodings(color,faces_loc) # encode the recognized faces

    for faceLoc , encodeFace in zip(faces_loc,faces_encode): # Make a loop for match each face location and encode face 
        matches = face_recognition.compare_faces(known_encode, encodeFace) # copare the new faces and known faces together
        face_distance = face_recognition.face_distance(known_encode, encodeFace) # find the distance of new encodes and known encodes
        matchesIndex = np.argmin(face_distance) # return the Minimum distance of faces


        if matches[matchesIndex]: # there is a condition that can improve the accuracy of recognition and if this condition will be true :
            name = names[matchesIndex].upper() # upper the names
            # print(name)

            y1,x2,y2,x1 = faceLoc # define the points of location of faces
            y1,x2,y2,x1 = y1*4 , x2*4 , y2*4 , x1*4 # because we resized the frame , now we have to multiply it by 4

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),2) # Draw a rectangle around the recognized faces
            cv2.rectangle(frame, (x1,y2-20), (x2,y2), (255,0,255),cv2.FILLED) # Draw a Filled rectangle
            cv2.putText(frame, name, (x1+4, y2-4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),1) # write down the name of face in the Filled rectangle that we has drawn

        else: # This Condition is for Unknown faces
            y1,x2,y2,x1 = faceLoc # define the points of location of faces
            y1,x2,y2,x1 = y1*4 , x2*4 , y2*4 , x1*4 # because we resized the frame , now we have to multiply it by 4

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),2) # Draw a rectangle around the recognized faces
            cv2.rectangle(frame, (x1,y2-20), (x2,y2), (255,0,255),cv2.FILLED) # Draw a Filled rectangle
            cv2.putText(frame, 'Unknown Face', (x1+4, y2-4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),1) # write down the name of face in the Filled rectangle that we has drawn


    cv2.imshow('output', frame) # show the output
    if cv2.waitKey(1) == ord('q'): # if you press q button the program will be closed
        break

#endregion


cap.release() # release the cap (webcam)
cv2.destroyAllWindows() # destroy all windows and close all of the resources

        
    
