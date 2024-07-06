import customtkinter as ctk
from utils import open_file_dialog


class Tabs(ctk.CTkTabview):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app

        # Create tabs
        self.add("Home")
        self.add("Image")
        self.add("Video")
        self.add("Webcam")

        # Home tab content
        home_title = ctk.CTkLabel(
            master=self.tab("Home"), text="Welcome to", font=("Arial", 40)
        )
        home_title.place(relx=0.5, rely=0.1, anchor="center")
        home_subtitle = ctk.CTkLabel(
            master=self.tab("Home"),
            text="DetectoBuddy",
            font=("Arial", 60),
            text_color="#007bff",
        )
        home_subtitle.place(relx=0.5, rely=0.2, anchor="center")
        home_description = ctk.CTkLabel(
            master=self.tab("Home"),
            text="This is a simple object detection application that can detect\nobjects in images, videos, and webcam\n\n\n\nUse the tabs to navigate the application.",
            font=("Arial", 20),
        )
        home_description.place(relx=0.5, rely=0.5, anchor="center")
        home_footer = ctk.CTkLabel(
            master=self.tab("Home"),
            text="Developed by: Luca Facchini (LF-D3v) & Avaab Razzaq (AR10Dev)",
            font=("Arial", 13),
        )
        home_footer.place(relx=0.5, rely=0.9, anchor="center")

        # Image tab content
        image_title = ctk.CTkLabel(
            master=self.tab("Image"), text="Detection from", font=("Arial", 30)
        )
        image_title.place(relx=0.5, rely=0.1, anchor="center")
        image_subtitle = ctk.CTkLabel(
            master=self.tab("Image"),
            text="Image",
            font=("Arial", 40),
            text_color="#007bff",
        )
        image_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        image_select_file_button = ctk.CTkButton(
            master=self.tab("Image"),
            text="Select Image",
            font=("Arial", 30),
            command=lambda: open_file_dialog("Image", self.app),
            width=150,
            height=18,
            corner_radius=10,
        )
        image_select_file_button.place(relx=0.5, rely=0.4, anchor="center")

        # Video tab content
        video_title = ctk.CTkLabel(
            master=self.tab("Video"), text="Detection from", font=("Arial", 30)
        )
        video_title.place(relx=0.5, rely=0.1, anchor="center")
        video_subtitle = ctk.CTkLabel(
            master=self.tab("Video"),
            text="Video",
            font=("Arial", 40),
            text_color="#007bff",
        )
        video_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        video_select_file_button = ctk.CTkButton(
            master=self.tab("Video"),
            text="Select Video",
            font=("Arial", 30),
            command=lambda: open_file_dialog("Video", self.app),
            width=150,
            height=18,
            corner_radius=10,
        )
        video_select_file_button.place(relx=0.5, rely=0.4, anchor="center")

        # Webcam tab content
        webcam_title = ctk.CTkLabel(
            master=self.tab("Webcam"), text="Detection from", font=("Arial", 30)
        )
        webcam_title.place(relx=0.5, rely=0.1, anchor="center")
        webcam_subtitle = ctk.CTkLabel(
            master=self.tab("Webcam"),
            text="Webcam",
            font=("Arial", 40),
            text_color="#007bff",
        )
        webcam_subtitle.place(relx=0.5, rely=0.2, anchor="center")

        webcam_start_button = ctk.CTkButton(
            master=self.tab("Webcam"),
            text="Start Webcam",
            font=("Arial", 30),
            command=lambda: self.app.webcam_detection(),
            width=150,
            height=18,
            corner_radius=10,
        )
        webcam_start_button.place(relx=0.5, rely=0.4, anchor="center")
