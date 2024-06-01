import tkinter as tk
import customtkinter as ctk
from ultralytics import YOLO
import cv2
import math
from tkinter import filedialog
from PIL import Image, ImageTk

ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("dark-blue")  


class ObjectDetector:
    def __init__(self, selected_class):
        self.model = YOLO("Yolo/yolov8n.pt")
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"
                           ]
        
        self.selected_class = selected_class

    def get_file_path(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        return file_path

    def draw_boxes(self, img, results):
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # put box in image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", self.classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(
                    img, self.classNames[cls], org, font, fontScale, color, thickness)

    def webcam(self, image_label):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        def update_frame():
            success, img = cap.read()
            if success:
                img = cv2.flip(img, 1)
                results = self.model(img, stream=True)
                self.draw_boxes(img, results)

                # Convert the image from BGR to RGB, then to PIL format and then to ImageTk format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the label
                image_label.configure(image=imgtk)
                image_label.image = imgtk

                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                else:
                    # Schedule the next update
                    image_label.after(10, update_frame)
            else:
                cap.release()
                cv2.destroyAllWindows()

        # Start the update process
        update_frame()

    def image(self, image_label):
        PATH = self.get_file_path()

        img = cv2.imread(PATH)
        results = self.model(img)
        self.draw_boxes(img, results)

        # Convert the image from BGR to RGB, then to PIL format and then to ImageTk format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label
        image_label.configure(image=imgtk)
        image_label.image = imgtk

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def video(self, image_label):
        PATH = self.get_file_path()

        cap = cv2.VideoCapture(PATH)

        while True:
            success, img = cap.read()

            if not success:
                break

            results = self.model(img)
            self.draw_boxes(img, results)

            # Convert the image from BGR to RGB, then to PIL format and then to ImageTk format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label
            image_label.configure(image=imgtk)
            image_label.image = imgtk

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_boxes(self, img, results):
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # class name
                cls = int(box.cls[0])
                class_name = self.classNames[cls]

                if class_name == self.selected_class:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # put box in image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    print("Confidence --->", confidence)

                    print("Class name -->", class_name)

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, class_name, org, font, fontScale, color, thickness)

def main():
    root = ctk.CTk()
    root.geometry("400x350")
    root.title("Object Detector")

    selected_class = "person"
    detector = ObjectDetector(selected_class)

    # Input Type Frame
    input_frame = ctk.CTkFrame(master=root, corner_radius=10)
    input_frame.pack(pady=20)

    input_label = ctk.CTkLabel(master=input_frame, text="Input Type:", font=("Roboto", 14))
    input_label.pack(side=tk.LEFT, padx=10)

    input_var = tk.StringVar(root)
    input_var.set("Webcam")
    input_options = ctk.CTkOptionMenu(master=input_frame, variable=input_var, values=["Webcam", "Image", "Video"])
    input_options.pack(side=tk.RIGHT, padx=10)

    # Class Selection Frame
    class_frame = ctk.CTkFrame(master=root, corner_radius=10)
    class_frame.pack(pady=10)

    class_label = ctk.CTkLabel(master=class_frame, text="Select Class:", font=("Roboto", 14))
    class_label.pack(side=tk.LEFT, padx=10)

    class_var = tk.StringVar(root)
    class_var.set(selected_class)
    class_options = ctk.CTkOptionMenu(master=class_frame, variable=class_var, values=detector.classNames)
    class_options.pack(side=tk.RIGHT, padx=10)

    # Image label
    image_label = ctk.CTkLabel(master=root)
    image_label.pack()

    # Start Button
    def start_detection():
        nonlocal detector
        selected_class = class_var.get()
        detector.selected_class = selected_class

        choice = input_var.get()

        if choice == 'Webcam':
            detector.webcam(image_label)
        elif choice == 'Image':
            detector.image(image_label)
        elif choice == 'Video':
            detector.video(image_label)

    start_button = ctk.CTkButton(master=root, text="Start Detection", command=start_detection, corner_radius=8, font=("Roboto", 14))
    start_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()