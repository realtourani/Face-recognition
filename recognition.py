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