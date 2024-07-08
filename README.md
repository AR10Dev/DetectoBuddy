# DetectoBuddy

<!-- <p align="center">
  <img src="logo.svg" alt="DetectoBuddy Logo" width="200"/>
</p> -->

DetectoBuddy is an advanced object detection application that utilizes the YOLO (You Only Look Once) model to identify objects in images, videos, and live webcam feeds. This project provides a user-friendly interface for effortless object detection across various media types.

## Features

- **Image Object Detection**: Upload and analyze images to detect objects.
- **Video Object Detection**: Process video files to identify objects in each frame.
- **Webcam Real-time Detection**: Utilize your webcam for live object detection.
- **User-friendly Interface**: Easy-to-use GUI built with [customtkinter](https://github.com/TomSchimansky/CustomTkinter).
- **Detailed Detection Information**: View confidence scores and bounding box coordinates for detected objects.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/AR10Dev/DetectoBuddy.git
   cd DetectoBuddy
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

## Usage

1. Launch the application by running `main.py`.
2. Use the tabs to switch between Image, Video, and Webcam detection modes.
3. For Image and Video modes, click "Select Image" or "Select Video" to choose a file.
4. Click "Detect now" to start the detection process.
5. For Webcam mode, click "Start Webcam" to begin real-time detection.

## Dependencies

- [ultralytics](https://github.com/ultralytics/ultralytics): For YOLO object detection
- [opencv-python-headless](https://github.com/opencv/opencv-python): For image and video processing
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter): For the modern GUI elements

## Contributing

Contributions to DetectoBuddy are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLO (You Only Look Once)](https://github.com/ultralytics/ultralytics) model by Ultralytics
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern GUI elements
- [OpenCV](https://opencv.org/) for computer vision capabilities

## Contact

If you have any questions, feel free to reach out to us or open an issue in the GitHub repository.

Happy detecting with DetectoBuddy! 🕵️‍♂️🔍
