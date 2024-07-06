import math
import os

import customtkinter as ctk
import cv2
import requests
from PIL import Image
from ultralytics import YOLO

from constants import classNames, IMG_PATH, VIDEO_PATH
from utils import back


def download_model_if_not_exists(
    model_url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt",
    model_folder="models",
    model_name="yolo.pt",
):
    """

    :param model_url:  (Default value = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt")
    :param model_folder:  (Default value = "models")
    :param model_name:  (Default value = "yolo.pt")

    """
    print("Starting the download process...")
    # Ensure the models folder exists
    os.makedirs(model_folder, exist_ok=True)
    print(f"Ensured {model_folder} folder exists.")

    model_path = os.path.join(model_folder, model_name)

    # Check if the model already exists
    if not os.path.isfile(model_path):
        print(f"{model_name} not found. Starting download...")
        # Download the model
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {model_name} successfully.")
        else:
            raise Exception(f"Failed to download {model_name}.")
    else:
        print(f"{model_name} already exists.")

    return model_path 


def image_detection(app):
    """

    :param app:

    """
    

    print("Starting image detection...")
    print(f"Image path: {IMG_PATH}")

    # check if the file still exists, or if the file got changed
    if not os.path.exists(IMG_PATH):
        app.image_error_path_header.configure(
            text="Image not found. Did it get deleted or moved?",
            text_color="red")
        app.image_path_header.configure(text="")
        app.image_path.configure(text="")
        app.image_detect_button.place_forget()
        return

    model_path = download_model_if_not_exists()

    # Load YOLO model
    model = YOLO(model_path)

    # Hide the main window
    app.withdraw()

    # Create a new window to display the detected image
    img_window = ctk.CTkToplevel()
    img_window.title("Detected Image")
    img_window.resizable(False, False)
    img_window.minsize(720, 1080)  # Set minimum size to 720x480
    img_window.maxsize(1240, 1780)  # Set maximum size to 1240x720

    # Create the "Go Back" button and place it at the top of the window, leaving some margin
    back_button = ctk.CTkButton(img_window, text="Go Back")
    back_button.pack(pady=10)

    # Create a text box for detections
    detections_textbox = ctk.CTkTextbox(img_window,
                                        width=80,
                                        height=20,
                                        font=("Arial", 15))
    detections_textbox.pack(side="top", fill="both", expand=True)

    # Insert a title
    detections_textbox.insert("end", "Detected Objects\n\n", "title")

    # Load the image
    img = cv2.imread(IMG_PATH)
    results = model(img)

    # Process the resized image with YOLO model and draw bounding boxes
    for i, r in enumerate(results):
        boxes = r.boxes
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color,
                        thickness)

    # Check if there are no detections after processing all results
    if not any(results):
        detections_textbox.insert("end", "No detections found.\n", "red")
    else:
        for i, r in enumerate(results):
            boxes = r.boxes
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                detections_textbox.insert("end", f"Detection {j + 1}:\n")
                detections_textbox.insert("end", f"Class: {classNames[cls]}\n")
                detections_textbox.insert("end", f"Confidence: {confidence}\n")
                detections_textbox.insert(
                    "end", f"Coordinates: ({x1}, {y1}) - ({x2}, {y2})\n\n")

    # Resize the image proportionally
    max_width = 1280
    max_height = 720
    width, height = img.shape[1], img.shape[0]
    aspect_ratio = width / height
    if aspect_ratio > max_width / max_height:
        width = max_width
        height = int(width / aspect_ratio)
    else:
        height = max_height
        width = int(height * aspect_ratio)

    img = cv2.resize(img, (width, height))

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_ctk = ctk.CTkImage(light_image=img_pil,
                           dark_image=img_pil,
                           size=(width, height))

    # Display the image in a Label and move it to the center of the window
    img_label = ctk.CTkLabel(img_window, image=img_ctk, text="")
    img_label.pack(fill="both", expand=True)  # Make the label fill the window

    # Keep a reference to the image to prevent garbage collection
    img_label.image = img_ctk

    # Bind the ButtonRelease event to the "Go Back" button and check if the mouse pointer is still over the button
    back_button.bind("<ButtonRelease-1>",
                     lambda event: back(event, img_window, None, None, app))


def video_detection(app):
    """

    :param app:

    """
    # Hide the main window
    app.withdraw()

    model_path = download_model_if_not_exists()

    # Load YOLO model
    model = YOLO(model_path)

    # Create a new window to display the detected video
    video_window = ctk.CTkToplevel()
    video_window.title("Detected Video")
    video_window.resizable(False, False)
    video_window.minsize(720, 1080)  # Set minimum size to 720x480
    video_window.maxsize(1240, 1780)  # Set maximum size to 1240x720

    # Create the "Go Back" button and place it at the top of the window, leaving some margin
    back_button = ctk.CTkButton(video_window, text="Go Back")
    back_button.pack(pady=10)

    # Initialize a variable to keep track of whether the video is paused
    paused = False

    # Create a text box for detections
    detections_textbox = ctk.CTkTextbox(video_window,
                                        width=80,
                                        height=20,
                                        font=("Arial", 15))
    detections_textbox.pack(side="top", fill="both", expand=True)

    # Insert a title
    detections_textbox.insert("end", "Detected Objects\n\n", "title")

    # Create a label to display the video
    video_label = ctk.CTkLabel(video_window)
    video_label.pack()

    # Create a video capture object
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Count the number of frames
    frame_number = 0

    # Display the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        frame_number += 1

        # Update the textbox
        detections_textbox.insert(
            "end", f"Detected objects in frame {frame_number}:\n")

        detected_objects = {
        }  # Dictionary to store detected objects in the frame

        for i, r in enumerate(results):
            boxes = r.boxes

            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, classNames[cls], org, font, fontScale,
                            color, thickness)

                cls = int(box.cls[0])
                class_name = classNames[cls]
                if class_name not in detected_objects:
                    detected_objects[class_name] = 1
                else:
                    detected_objects[class_name] += 1

        # Append detected objects to the message
        for obj, count in detected_objects.items():
            detections_textbox.insert("end", f"{count} {obj}, ")

        # Move to the next line after appending all detected objects
        detections_textbox.insert("end", "\n\n")

        # Auto-scroll the textbox
        detections_textbox.yview_moveto(1.0)

        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the image to PIL format
        img = Image.fromarray(frame)

        # Convert the image to CTkImage format
        img_ctk = ctk.CTkImage(light_image=img,
                               dark_image=img,
                               size=(img.width, img.height))

        video_label.configure(image=img_ctk, text="")
        video_label.image = img_ctk

        # Update the window
        video_window.update()

        # Bind the ButtonRelease event to the "Go Back" button
        back_button.bind(
            "<ButtonRelease-1>",
            lambda event: back(event, video_window, "video", cap, app),
        )
    # Kill the window
    video_window.destroy()


def webcam_detection(app):
    """

    :param app:

    """
    # close the main window
    app.withdraw()

    # Create a new window to display the webcam feed
    webcam_window = ctk.CTkToplevel()
    webcam_window.title("Webcam Feed")
    webcam_window.resizable(False, False)
    webcam_window.minsize(720, 720)

    # Create the "Go Back" button and place it at the top of the window, leaving some margin
    back_button = ctk.CTkButton(webcam_window, text="Go Back")
    back_button.pack(pady=10)

    # Create a text box for detections
    detections_textbox = ctk.CTkTextbox(webcam_window,
                                        width=80,
                                        height=20,
                                        font=("Arial", 15))
    detections_textbox.pack(side="top", fill="both", expand=True)

    # Insert a title
    detections_textbox.insert("end", "Detected Objects\n\n", "title")

    # Create a label to display the webcam feed
    webcam_label = ctk.CTkLabel(webcam_window)
    webcam_label.pack()

    # Initialize the webcam and model
    cap = cv2.VideoCapture(0)
    
    model_path = download_model_if_not_exists()

    # Load YOLO model
    model = YOLO(model_path)

    while True:
        success, img = cap.read()

        # Resize the frame to fit the window
        img = cv2.resize(img, (720, 480))

        # Invert the frame
        inverted_img = cv2.flip(img, 1)
        results = model(inverted_img, stream=True)

        # Coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )  # Convert to int values

                # Put box in cam
                cv2.rectangle(inverted_img, (x1, y1), (x2, y2), (255, 0, 255),
                              3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Class name
                cls = int(box.cls[0])

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(
                    inverted_img,
                    classNames[cls],
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                )

                detections_textbox.yview_moveto(1.0)

                # Update the textbox
                detections_textbox.insert(
                    "end",
                    f"Detected {classNames[cls]} with confidence {confidence}\n"
                )

        # Convert the image from BGR to RGB
        inverted_img = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2RGB)

        # Convert the image to PIL format
        img_pil = Image.fromarray(inverted_img)

        # Convert the image to CTkImage format
        img_ctk = ctk.CTkImage(
            light_image=img_pil,
            dark_image=img_pil,
            size=(img_pil.width, img_pil.height),
        )

        webcam_label.configure(image=img_ctk, text="")
        webcam_label.image = img_ctk

        # Update the window
        webcam_window.update()

        # Bind the ButtonRelease event to the "Go Back" button
        back_button.bind(
            "<ButtonRelease-1>",
            lambda event: back(event, webcam_window, "webcam", cap, app),
        )

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
