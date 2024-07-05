import os
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog

def open_file_dialog(selection, app):
    global IMG_PATH
    global VIDEO_PATH

    if selection == 'Image':
        IMG_PATH = filedialog.askopenfilename()
        print_file_path(selection, IMG_PATH, app)
    elif selection == 'Video':
        VIDEO_PATH = filedialog.askopenfilename()
        print_file_path(selection, VIDEO_PATH, app)

def print_file_path(selection, file_path, app):
    if selection == 'Image':
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