import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
import constants

def open_file_dialog(selection, app):
    if selection == 'Image':
        constants.IMG_PATH = filedialog.askopenfilename()
        print_file_path(selection, app)
    elif selection == 'Video':
        constants.VIDEO_PATH = filedialog.askopenfilename()
        print_file_path(selection, app)

def print_file_path(selection, app):
    IMG_PATH = constants.IMG_PATH
    VIDEO_PATH = constants.VIDEO_PATH
       
    if selection == 'Image':
        if IMG_PATH.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            app.image_path_header.configure(text="Currently selected image", text_color="green")
            app.image_error_path_header.configure(text="")
            app.image_path.configure(text=IMG_PATH)
            app.image_detect_button.place(relx=0.5, rely=0.85, anchor="center")
        elif IMG_PATH == "":
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
        if VIDEO_PATH.endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv')):
            app.video_path_header.configure(text="Currently selected video", text_color="green")
            app.video_error_path_header.configure(text="")
            app.video_path.configure(text=VIDEO_PATH)
            app.video_detect_button.place(relx=0.5, rely=0.85, anchor="center")
        elif VIDEO_PATH == "":
            app.video_error_path_header.configure(text="No file selected", text_color="red")
            app.video_path_header.configure(text="")
            app.video_path.configure(text="")
            app.video_detect_button.place_forget()
        else:
            app.video_error_path_header.configure(text="Invalid file format", text_color="red")
            app.video_path_header.configure(text="")
            app.video_path.configure(text="")
            app.video_detect_button.place_forget()

def cv2_to_imagetk(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(image)
    imageTk = ImageTk.PhotoImage(image=imagePIL)
    return imageTk

def back(event, window, type, cap, app):
    if window.winfo_containing(event.x_root, event.y_root) == event.widget:
        if type == "video":
            cap.release()
            app.deiconify()
            return
        if type == "webcam":
            cap.release()
        app.deiconify()
        window.destroy()