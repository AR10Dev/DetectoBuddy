import customtkinter
from customtkinter import *
from ultralytics import YOLO
import cv2
import math
from PIL import Image, ImageTk
import PIL

IMG_PATH = None
VIDEO_PATH = None
MODEL_PATH = "Yolo/yolov8n.pt"
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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


class AutoDetectorApp(CTk):
    def __init__(self):
        super().__init__()

        # Set window title and disable resizing
        self.video_detect_button = None
        self.video_path = None
        self.video_error_path_header = None
        self.video_path_header = None
        self.image_detect_button = None
        self.image_path = None
        self.image_error_path_header = None
        self.image_path_header = None
        self.title("AutoDetector v1.0.0")
        self.resizable(False, False)

        # Create tab view
        self.tab_view = Tabs(master=self, width=800, height=600)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20)

        # Initialize Image and Video tab elements
        self.initialize_image_tab_elements()
        self.initialize_video_tab_elements()

    def initialize_image_tab_elements(self):
        self.image_path_header = CTkLabel(master=self.tab_view.tab("Image"), text="", font=("Arial", 20))
        self.image_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.image_error_path_header = CTkLabel(master=self.tab_view.tab("Image"), text="", font=("Arial", 20),
                                                text_color="red")
        self.image_error_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.image_path = CTkLabel(master=self.tab_view.tab("Image"), text="", font=("Arial", 15))
        self.image_path.place(relx=0.5, rely=0.65, anchor="center")

        self.image_detect_button = CTkButton(master=self.tab_view.tab("Image"), text="Detect now", font=("Arial", 30),
                                             command=lambda: image_detection(), width=150, height=18,
                                             corner_radius=10)
        self.image_detect_button.place_forget()

    def initialize_video_tab_elements(self):
        self.video_path_header = CTkLabel(master=self.tab_view.tab("Video"), text="", font=("Arial", 20))
        self.video_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.video_error_path_header = CTkLabel(master=self.tab_view.tab("Video"), text="", font=("Arial", 20),
                                                text_color="red")
        self.video_error_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.video_path = CTkLabel(master=self.tab_view.tab("Video"), text="", font=("Arial", 15))
        self.video_path.place(relx=0.5, rely=0.65, anchor="center")

        self.video_detect_button = CTkButton(master=self.tab_view.tab("Video"), text="Detect now", font=("Arial", 30),
                                             command=lambda: print("g"), width=150, height=18,
                                             corner_radius=10)
        self.video_detect_button.place_forget()


class Tabs(CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Create tabs
        self.add("Home")
        self.add("Image")
        self.add("Video")
        self.add("Webcam")

        # Home tab content
        home_title = CTkLabel(master=self.tab("Home"), text="Welcome to", font=("Arial", 40))
        home_title.place(relx=0.5, rely=0.1, anchor="center")
        home_subtitle = CTkLabel(master=self.tab("Home"), text="AutoDetector", font=("Arial", 60), text_color="#007bff")
        home_subtitle.place(relx=0.5, rely=0.2, anchor="center")
        home_description = CTkLabel(master=self.tab("Home"),
                                    text="This is a simple object detection application that can detect\nobjects in "
                                         "images, videos, and webcam\n\n\n\nUse the tabs to navigate the application.",
                                    font=("Arial", 20))
        home_description.place(relx=0.5, rely=0.5, anchor="center")
        home_footer = CTkLabel(master=self.tab("Home"),
                               text="Developed by: Luca Facchini (LF-D3v) & Avaab Razzaq (AR10Dev)", font=("Arial", 13))
        home_footer.place(relx=0.5, rely=0.9, anchor="center")

        # Image tab content
        image_title = CTkLabel(master=self.tab("Image"), text="Detection from", font=("Arial", 30))
        image_title.place(relx=0.5, rely=0.1, anchor="center")
        image_subtitle = CTkLabel(master=self.tab("Image"), text="Image", font=("Arial", 40), text_color="#007bff")
        image_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        image_select_file_button = CTkButton(master=self.tab("Image"), text="Select Image", font=("Arial", 30),
                                             command=lambda: open_file_dialog('Image'), width=150, height=18,
                                             corner_radius=10)
        image_select_file_button.place(relx=0.5, rely=0.4, anchor="center")

        # Video tab content
        video_title = CTkLabel(master=self.tab("Video"), text="Detection from", font=("Arial", 30))
        video_title.place(relx=0.5, rely=0.1, anchor="center")
        video_subtitle = CTkLabel(master=self.tab("Video"), text="Video", font=("Arial", 40), text_color="#007bff")
        video_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        video_select_file_button = CTkButton(master=self.tab("Video"), text="Select Video", font=("Arial", 30),
                                             command=lambda: open_file_dialog('Video'), width=150, height=18,
                                             corner_radius=10)
        video_select_file_button.place(relx=0.5, rely=0.4, anchor="center")


def open_file_dialog(selection):
    global IMG_PATH
    if selection == 'Image':
        IMG_PATH = filedialog.askopenfilename()
        print_file_path(selection, IMG_PATH)
    elif selection == 'Video':
        VIDEO_PATH = filedialog.askopenfilename()
        print_file_path(selection, VIDEO_PATH)


def print_file_path(selection, file_path):
    if selection == 'Image':
        # check if the file format is valid.
        if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            app.image_path_header.configure(text="Currently selected image", text_color="green")
            app.image_error_path_header.configure(text="")

            app.image_path.configure(text=file_path)
            app.image_detect_button.place(relx=0.5, rely=0.85, anchor="center")
        elif file_path == "":
            app.image_error_path_header.configure(text="No file selected", text_color="red")
            app.image_path_header.configure(text="")
            app.image_path.configure(text="")
            app.image_detect_button.place_forget()
        else:
            app.image_error_path_header.configure(text="Invalid file format", text_color="red")
            app.image_path_header.configure(text="")
            app.image_path.configure(text="")
            app.image_detect_button.place_forget()

    elif selection == 'Video':
        # check if the file format is valid.
        if file_path.endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv')):
            app.video_path_header.configure(text="Currently selected video", text_color="green")
            app.video_error_path_header.configure(text="")

            app.video_path.configure(text=file_path)
            app.video_detect_button.place(relx=0.5, rely=0.85, anchor="center")
        elif file_path == "":
            app.video_error_path_header.configure(text="No file selected", text_color="red")
            app.video_path_header.configure(text="")
            app.video_path.configure(text="")
            app.video_detect_button.place_forget()
        else:
            app.video_error_path_header.configure(text="Invalid file format", text_color="red")
            app.video_path_header.configure(text="")
            app.video_path.configure(text="")
            app.video_detect_button.place_forget()


# So far, this function will ONLY display the image in a new window. We will implement the detection part later.
def cv2_to_imagetk():
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = PIL.Image.fromarray(image)
    imageTk = ImageTk.PhotoImage(image = imagePIL)

    return imageTk
def image_detection():
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Hide the main window
    app.withdraw()

    # Create a new window to display the detected image
    img_window = customtkinter.CTkToplevel()
    img_window.title("Detected Image")
    img_window.resizable(False, False)
    img_window.minsize(720, 480)  # Set minimum size to 720x480
    img_window.maxsize(1240, 720)  # Set maximum size to 1240x720

    # Create the "Go Back" button and place it at the top of the window, leaving some margin
    back_button = customtkinter.CTkButton(img_window, text="Go Back")
    back_button.pack(pady=10)

    # Load the image
    img = cv2.imread(IMG_PATH)
    results = model(img)

    # Process the resized image with YOLO model and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Convert the image to a format that can be displayed in CustomTkinter
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    while img_pil.width > 1240 or img_pil.height > 720:
        img_pil = img_pil.resize((int(img_pil.width / 1.1), int(img_pil.height / 1.1)))

    while img_pil.width < 720 or img_pil.height < 480:
        img_pil = img_pil.resize((int(img_pil.width * 1.1), int(img_pil.height * 1.1)))

    img_ctk = CTkImage(light_image=img_pil, dark_image=img_pil, size=(img_pil.width, img_pil.height))

    # if the image is bigger or smaller than the window, resize it to fit the window

    # Display the image in a Label and move it to the center of the window
    img_label = customtkinter.CTkLabel(img_window, image=img_ctk, text="")
    img_label.pack(fill="both", expand=False)  # Make the label fill the window

    # Keep a reference to the image to prevent garbage collection
    img_label.image = img_ctk

    # Bind the ButtonRelease event to the "Go Back" button and check if the mouse pointer is still over the button
    back_button.bind("<ButtonRelease-1>", lambda event: back(event, img_window))

def back(event, window):
    if window.winfo_containing(event.x_root, event.y_root) == event.widget:
        window.destroy()
        app.deiconify()

app = AutoDetectorApp()
app.mainloop()
