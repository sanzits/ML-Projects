import json
import cv2
import mediapipe as mp
import os
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
#IMAGE_FILES = ["input.jpeg"]

def detect_faces(input_files_list, folder_path):

    face_map = {} #dictionary to store mapping of file names to detected faces
    #creates a face detection object
    #keyword "with" ensures that Mediapipe cleans up GPU/CPU resources automatically
    
    with mp_face_detection.FaceDetection(
        #model_selection = 0: short-range model, best for faces within 2 meters
        #model_selection = 1: full-range model, best for faces beyond 2 meters
        #min_detection_confidence = minimum confidence value ([0.0, 1.0]) for face detection to be considered successful
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, file in enumerate(input_files_list):
            #cv2.imread(file): loads the actual image into a NumPy array
            image = cv2.imread(f"{folder_path}/{file}")
            image_path = os.path.join(folder_path, file)
            face_map[file] = {
                "full_path": image_path,
                "cropped_faces": []
                }
            #OpenCV uses BGR
            #Mediapipe expects RGB
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                continue
            #We don’t draw on the original; we draw on a duplicate.
            
            annotated_image = image.copy()
            for face_idx, detection in enumerate(results.detections):
            # This prints the relative coordinates of the nose.
                #print('Nose tip:')
                print("file_name:", file, " face_index:", face_idx)
                print(mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                    # Mediapipe draws:
                    # Face bounding box
                    # Confidence score
                    # Key points (nose, eyes, mouth corners)
                mp_drawing.draw_detection(annotated_image, detection)
                # Crop the face using the bounding box
                bbox = detection.location_data.relative_bounding_box   
                h, w, _= image.shape
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                box_width = int(bbox.width * w)
                box_height = int(bbox.height * h)

                # Ensure the bounding box is within image boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_min + box_width)
                y_max = min(h, y_min + box_height)

                face_crop = image[y_min:y_max, x_min:x_max]
                #write the siamese cnn function here to find out who is this person.
                face_file = f'/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face Detection/face_crops/cropped_face_{idx}_{face_idx}.png'
                cv2.imwrite(str(face_file), face_crop)
                face_map[file]["cropped_faces"].append(face_file)
                print(f"Saved cropped face to {face_file}")
                #cv2.imwrite(f'/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face Detection/annotated_image_{idx}_{face_idx}.png', annotated_image)
    return face_map


folder_path = "/Users/sanchitsuman/Documents/Face recognition/Input"
input_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpeg",".jpg"))]
print(input_files)
face_map = detect_faces(input_files, folder_path)
with open("/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face Detection/face_mapping/face_map.json", "w") as f:
    json.dump(face_map, f, indent=4)
