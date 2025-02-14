from deepface import DeepFace
import os
import cv2

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]


def check_folder(pic):
    count_files = 0
    count_success = 0
    count_number_people = 0
    success_count = 0

    for person2 in os.listdir('Database'):
        # print(person2.split("-"))
        person = person2.split("-")[0]
        p = person2.split("-")[1]
        # print(person)
        count_number_people += 1
        for person_pictures in os.listdir(os.path.join('Database', person2)):
            count_files += 1
            # print(person_pictures, person2)
            data = DeepFace.verify(pic, 'Database/' + str(person) + "-" + p + '/Pic_' + str(count_files) + '.jpg', model_name = models[2])
            # print(data)
            if data["distance"]<0.4:
                count_success +=1
                # data = DeepFace.verify(pic, 'Database/' + str(person) + '/Pic_' + str(count_files) + '.jpg')
                # print ("Success")
                # print (data)
                # print (count_success)

        # print("\n\n",count_success, 0.4*count_files)
        if count_success>=0.4*count_files and count_files<10:
            return p
            
            
        count_files=0
        count_success=0
        
    if success_count==0:
        return "Stranger"
 

    

    