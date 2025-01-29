import dlib
import cv2
import numpy as np

# Load Dlib models
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1("/Users/shauryabhardwaj/Desktop/programs/2025/images_aarav_thing/dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("/Users/shauryabhardwaj/Desktop/programs/2025/images_aarav_thing/shape_predictor_68_face_landmarks.dat")

# Function to extract face descriptor from an image
def get_face_descriptor(image_path):
    image = cv2.imread(image_path)
    face_locations = face_detector(image, 1)

    if len(face_locations) > 0:
        face_descriptor = face_encoder.compute_face_descriptor(image, shape_predictor(image, face_locations[0]))
        return np.array(face_descriptor)
    return None

# Load your images and extract the descriptors
image_path_1 = '/Users/shauryabhardwaj/Desktop/programs/2025/images_aarav_thing/IMG_0779.JPG'
image_path_2 = '/Users/shauryabhardwaj/Desktop/programs/2025/images_aarav_thing/f775084b-a5b0-4b45-b3c3-3326d666d0cc.JPG'

shaura_face_descriptor_1 = get_face_descriptor(image_path_1)
shaura_face_descriptor_2 = get_face_descriptor(image_path_2)

# If face descriptors are found, average them to get a final reference descriptor
if shaura_face_descriptor_1 is not None and shaura_face_descriptor_2 is not None:
    shaura_face_descriptor = (shaura_face_descriptor_1 + shaura_face_descriptor_2) / 2
else:
    print("Error: No faces detected in the provided images. Exiting...")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to initialize webcam and check if it's accessible
def initialize_webcam(camera_index=0):
    cap = cv2.VideoCapture(camera_index)  # Try different camera indices if necessary
    if not cap.isOpened():
        print(f"Failed to access camera at index {camera_index}. Trying another index...")
        return None
    return cap

# Try initializing the webcam
cap = initialize_webcam(0)  # Default camera
if cap is None:
    cap = initialize_webcam(1)  # Try with camera index 1 if index 0 fails
    if cap is None:
        print("Unable to access the camera. Exiting...")
        exit()

# Real-time face comparison with the reference face (Shaurya)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect faces in the frame
    face_locations = face_detector(frame, 1)
    
    # Process faces
    if len(face_locations) > 0:
        for face_location in face_locations:
            # Draw rectangle around the face
            (x, y, w, h) = (face_location.left(), face_location.top(), face_location.width(), face_location.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract descriptor for the live frame
        face_descriptors_1 = [face_encoder.compute_face_descriptor(frame, shape_predictor(frame, face_location)) for face_location in face_locations]
        live_face_descriptor = np.array(face_descriptors_1[0])

        # Calculate the distance between Shaurya's reference face and the live face
        face_distance = np.linalg.norm(shaura_face_descriptor - live_face_descriptor)

        # Label the face based on the comparison
        if face_distance < 0.6:
            label = "Shaurya"
        else:
            label = "Not Shaurya"

        # Add the label text to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) if label == "Shaurya" else (0, 0, 255)
        cv2.putText(frame, label, (50, 50), font, 1, color, 2, cv2.LINE_AA)

        # Print the label to console
        print(label)

    # Show the frame with detected faces and label
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
