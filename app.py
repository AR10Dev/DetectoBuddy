import customtkinter as ctk
from tabs import Tabs
import os
import requests
import tempfile


class AutoDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Download YOLO model
        self.download_yolo_model()

        # Set window title and disable resizing
        self.title("AutoDetector v1.0.0")
        self.resizable(False, False)

        # Create tab view
        self.tab_view = Tabs(master=self, width=800, height=600, app=self)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20)

        # Initialize Image and Video tab elements
        self.initialize_image_tab_elements()
        self.initialize_video_tab_elements()

    def initialize_image_tab_elements(self):
        self.image_path_header = ctk.CTkLabel(
            master=self.tab_view.tab("Image"), text="", font=("Arial", 20))
        self.image_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.image_error_path_header = ctk.CTkLabel(master=self.tab_view.tab(
            "Image"), text="", font=("Arial", 20), text_color="red")
        self.image_error_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.image_path = ctk.CTkLabel(master=self.tab_view.tab(
            "Image"), text="", font=("Arial", 15))
        self.image_path.place(relx=0.5, rely=0.65, anchor="center")

        self.image_detect_button = ctk.CTkButton(master=self.tab_view.tab("Image"), text="Detect now", font=("Arial", 30),
                                                 command=lambda: self.image_detection(), width=150, height=18, corner_radius=10)
        self.image_detect_button.place_forget()

    def initialize_video_tab_elements(self):
        self.video_path_header = ctk.CTkLabel(
            master=self.tab_view.tab("Video"), text="", font=("Arial", 20))
        self.video_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.video_error_path_header = ctk.CTkLabel(master=self.tab_view.tab(
            "Video"), text="", font=("Arial", 20), text_color="red")
        self.video_error_path_header.place(relx=0.5, rely=0.6, anchor="center")

        self.video_path = ctk.CTkLabel(master=self.tab_view.tab(
            "Video"), text="", font=("Arial", 15))
        self.video_path.place(relx=0.5, rely=0.65, anchor="center")

        self.video_detect_button = ctk.CTkButton(master=self.tab_view.tab("Video"), text="Detect now", font=("Arial", 30),
                                                 command=lambda: self.video_detection(), width=150, height=18, corner_radius=10)
        self.video_detect_button.place_forget()

    def image_detection(self):
        from detection import image_detection
        image_detection(self)

    def video_detection(self):
        from detection import video_detection
        video_detection(self)

    def webcam_detection(self):
        from detection import webcam_detection
        webcam_detection(self)
