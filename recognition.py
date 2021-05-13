#region imports
import cv2
import face_recognition
import numpy as np
import os
#endregion

#region variable
path = 'images'
names = []
array_images = []
my_list = []
my_list = os.listdir(path)
# print(my_list)
#endregion

#region known image
for item in my_list:

    current_img = cv2.imread(f'{path}/{item}')
    array_images.append(current_img)
    names.append(item.split('.')[0])

# print(names)
# print(array_images)

#endregion

#region known encoding
def encoding(array_img):
    encode_list = []
    for image in array_images:

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)

    return encode_list

known_encode = encoding(array_images)
# print(len(known_encode))
# print('---- done ----')

#endregion

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    resize = cv2.resize(frame, (0,0),None,0.25,0.25)
    resize = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
    faces_loc = face_recognition.face_locations(resize)
    faces_encode = face_recognition.face_encodings(resize,faces_loc)
    
    for encodeFace, faceLoc in zip(faces_encode,faces_loc):
        matches = face_recognition.compare_faces(known_encode, encodeFace)
        face_distance = face_recognition.face_distance(known_encode, encodeFace)
        matchesIndex = np.argmin(face_distance)


        if matches[matchesIndex]:
            name = names[matchesIndex].upper()
            # print(name)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4 , x2*4 , y2*4 , x1*4

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),2)
            cv2.rectangle(frame, (x1,y2-15), (x2,y2), (255,0,255),cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),1)


    cv2.imshow('output', frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()