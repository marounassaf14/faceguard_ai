import cv2
import os
from mtcnn import MTCNN

def crop_faces(input_folder, output_folder, start_index=0):
    os.makedirs(output_folder, exist_ok=True)
    detector = MTCNN()

    # Existing cropped faces
    existing_files = [f for f in os.listdir(output_folder) if f.endswith((".jpg", ".png"))]
    face_start_index = len(existing_files)

    count = 0
    files = sorted(os.listdir(input_folder))
    new_files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")][start_index:]

    for file in new_files:
        image_path = os.path.join(input_folder, file)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)

        if results:
            x, y, width, height = results[0]['box']
            x, y = max(x, 0), max(y, 0)
            cropped_face = image[y:y+height, x:x+width]

            save_path = os.path.join(output_folder, f"face_{face_start_index + count:04d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            count += 1

    return count
