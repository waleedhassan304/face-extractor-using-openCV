#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab.patches import cv2_imshow
import cv2
import dlib
import os

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Input folder path containing images
input_folder = 'path to input folder/'

# Output folder path for saving detected faces
output_folder = 'path to output folder/processed/'

def MyRec(rgb, x, y, w, h, v=20, color=(200, 0, 0), thickness=2):
    """To draw a stylish rectangle around the objects"""
    cv2.line(rgb, (x, y), (x + v, y), color, thickness)
    cv2.line(rgb, (x, y), (x, y + v), color, thickness)
    cv2.line(rgb, (x + w, y), (x + w - v, y), color, thickness)
    cv2.line(rgb, (x + w, y), (x + w, y + v), color, thickness)
    cv2.line(rgb, (x, y + h), (x, y + h - v), color, thickness)
    cv2.line(rgb, (x, y + h), (x + v, y + h), color, thickness)
    cv2.line(rgb, (x + w, y + h), (x + w, y + h - v), color, thickness)
    cv2.line(rgb, (x + w, y + h), (x + w - v, y + h), color, thickness)

def save(img, name, bbox, width=224, height=224, padding=(20, 30)):  # Adjusted padding for top side
    x, y, w, h = bbox
    # Expand the bounding box with different padding values
    x -= padding[0] * 3  # Padding on the left
    y -= padding[1] * 4  # Padding on the top
    w += padding[0] * 3  # Padding on the left and right
    h += padding[1] * 2  # Padding on the top and bottom
    # Ensure the coordinates are within bounds
    x = max(x, 0)
    y = max(y, 0)
    imgCrop = img[y:h, x:w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name + ".jpg", imgCrop)

def extract_faces_from_folder(folder_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through the images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)

            # Load the image
            frame = cv2.imread(image_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Detect faces in the image
            for counter, face in enumerate(faces):
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 255, 220), 1)
                MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0, 250, 0), 3)
                save(gray, os.path.join(output_folder, f'{filename}_{counter}'), (x1, y1, x2, y2), padding=(20, 50))

            # Display the processed image
            frame = cv2.resize(frame, (800, 800))
            cv2_imshow(frame)
            cv2.waitKey(0)
    
    print("Done saving")

# Call the function to extract faces from the input folder
extract_faces_from_folder(input_folder)


# In[ ]:


from google.colab.patches import cv2_imshow
import cv2
import os

# Path to the folder containing images
folder_path = 'path to /processed'

# Iterate through the images in the folder and display each one
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)

        # Load and display the image
        image = cv2.imread(image_path)
        print("Displaying:", filename)
        cv2_imshow(image)

# Wait for a key press to close the displayed images
cv2.waitKey(0)
cv2.destroyAllWindows()

