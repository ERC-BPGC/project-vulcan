import face_recognition
import cv2
import numpy as np
import math
import mediapipe as mp

print("Starting...")

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

faces_cur_frame = []
face_encodings = []
faces = {}
encodings_cur_frame = []
face_landmarks = []
frame_count = 0
speakers = []
cur_speakers = []
thresh = None

width = 640
height = 480

point_list = [39, 181, 37, 84, 0, 17, 267, 314, 269, 405,
              81, 178, 82, 87, 13, 14, 312, 317, 311, 402]


def distance(landmarks):
    p_lst = []
    dist = 0

    for i in point_list:
        p_lst.append([landmarks.landmark[i].x*width, landmarks.landmark[i].y*height])

    for i in range(0, len(p_lst)-1, 2):
        p = p_lst[i]
        p1 = p_lst[i+1]

        val = math.sqrt((p1[0]-p[0])**2 + (p1[1]-p[1])**2)
        dist += val

    return dist


def reference(landmarks):
    lft = landmarks.landmark[61] 
    rt = landmarks.landmark[291] 
    ref = math.sqrt(((rt.x*width)-(lft.x*width))**2 + ((rt.y*height)-(lft.y*height))**2)
    return ref


with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=3) as face_mesh:
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while True:
            success, frame = cap.read()

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results_mesh = face_mesh.process(image)
            results_faces = face_detection.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_count += 1
            if frame_count == 5:
                process = True
                frame_count = 0
                print("faces: ", faces)
                cur_speakers = []
            else:
                process = False


            faces_cur_frame = []
            encodings_cur_frame = []
            face_landmarks_mp = results_mesh.multi_face_landmarks

            try:
                for detection in results_faces.detections:
                    bbox = detection.location_data.relative_bounding_box
                    top = int(bbox.ymin*height)
                    right = int((bbox.xmin + bbox.width)*width)
                    bottom = int((bbox.ymin + bbox.height)*height)
                    left = int(bbox.xmin*width)
                    faces_cur_frame.append((top, right, bottom, left))
                    crop = image[top:bottom, left:right]
                    encoding = face_recognition.face_encodings(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if encoding:
                        encodings_cur_frame.append(encoding[0])
            except:
                pass

            num_faces = len(encodings_cur_frame)

            if len(face_encodings) == 0:
                for i in range(num_faces):
                    faces[i] = {"val": 0,
                                "ref": 0}

                    face_encodings.append(encodings_cur_frame[i])
                    print("faces added")

            else:
                for i, face_encoding, (top, right, bottom, left) in zip(range(num_faces), encodings_cur_frame, faces_cur_frame):
                    mid = (left + (right-left)//2, top+ (bottom-top)//2)
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

                    if i in speakers:
                        cv2.rectangle(image, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(image, 'speaking', (left + 6, bottom - 5), font, 0.5, (255, 255, 255), 2)

                    dist = 1000000
                    val = None

                    val = [distance(face_landmarks_mp[i]), reference(face_landmarks_mp[i])]

                    matches = face_recognition.compare_faces(face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        faces[best_match_index]["val"] += val[0]
                        faces[best_match_index]["ref"] += val[1]

                        if process:
                            ref = faces[best_match_index]["ref"] 
                            gain = (-0.00142857142857*ref) + (2.15285714286)
                            if faces[best_match_index]["val"] > (ref*gain):
                                cur_speakers.append(i)
                                print(i)

                            faces[best_match_index]["val"] = 0
                            faces[best_match_index]["ref"] = 0

                    else:
                        if process:
                            faces[len(faces)] = {"val": 0,
                                                "ref": 0}
                        else:
                            faces[len(faces)] = {"val": val[0],
                                                "ref": val[1]}
                        face_encodings.append(face_encoding)
                        print("face_added")

            if process:
                speakers = cur_speakers
                
            cv2.imshow('Webcam', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

print('Program Terminated')
