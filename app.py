import datetime
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import sv_ttk  # install this: pip install sv-ttk
import os

class UtensilManagementApp:
    def __init__(self, root):
        # Initialize the main application
        self.root = root
        self.root.title('Automated Utensil Management System')
        sv_ttk.use_dark_theme()  # theme
        self.label_font = font.Font(family="Lucida Console", size=15, weight="bold")

        # screen dimension
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.root.minsize(640, 720)
        img = tk.Image("photo", file="logo.gif")
        self.root.tk.call('wm','iconphoto', self.root._w, img)

        # Create YOLO model and necessary variables
        self.create_directory('Yolo-Weights')
        self.model = YOLO('Yolo-Weights/yolov8m.pt')
        self.classes = self.load_classes("classes.txt")
        self.class_colors = np.random.randint(0, 255, (len(self.classes), 3))

        # Placement and color mapping dictionaries
        self.placement = {
            "fork_x": -1,
            "knife_x": -1,
            "spoon_x": -1
        }

        self.color_mapping = {
            "fork": (112, 149, 91),  # Green
            "knife": (242, 165, 156),  # Red
            "spoon": (100, 175, 168)  # Blue
        }

        # Open video capture and initialize other variables
        self.cap = cv2.VideoCapture(0)
        self.out = None

        self.is_detection_started = False
        self.is_recording_started = False

        # Create GUI widgets
        self.create_widgets()
        self.configure_layout()

    # Function to create a directory if it doesn't exist
    def create_directory(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")

    # Function to load classes from a file
    def load_classes(self, file_path):
        with open(file_path, "r") as file:
            return [class_name.strip() for class_name in file.readlines()]

    # Function to create GUI widgets
    def create_widgets(self):
        self.top_frame = tk.Frame(self.root, borderwidth=1, relief="flat", bg="#2F2F2F")
        self.text_frame = tk.Frame(self.root)
        self.bottom_frame = tk.Frame(self.root, borderwidth=1, relief="flat", bg="#2F2F2F")

        self.camera = ttk.Label(self.top_frame)
        self.b1 = ttk.Button(self.bottom_frame, text='START\nUTENSIL DETECTION 🔎', command=self.toggle_detection)
        self.b2 = ttk.Button(self.bottom_frame, text='RECORD 🔴', command=self.toggle_recording)

    # Function to configure GUI layout
    def configure_layout(self):
        self.top_frame.place(x=0, y=0, relwidth=1, relheight=0.9)
        self.text_frame.place(x=0, rely=0.9, relwidth=1, relheight=0.05)
        self.bottom_frame.place(x=0, rely=0.95, relwidth=1, relheight=0.05)

        # Top Frame Layout
        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_rowconfigure(0, weight=1)
        self.camera.grid(row=0, column=0, sticky="ns")

        # Bottom Frame Layout
        self.bottom_frame.grid_columnconfigure((0, 1), weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.b1.grid(row=0, column=0, sticky="nwes")
        self.b2.grid(row=0, column=1, sticky="nwes")

    # Function to toggle utensil detection
    def toggle_detection(self):
        if not self.is_detection_started:
            self.is_detection_started = True
            self.b1["text"] = 'STOP\nUTENSIL DETECTION 🔎'
        else:
            self.is_detection_started = False
            self.b1["text"] = 'START\nUTENSIL DETECTION 🔎'
            for widget in self.text_frame.winfo_children():
                widget.destroy()

    # Function to toggle video recording
    def toggle_recording(self):
        if not self.is_recording_started:
            self.is_recording_started = True
            self.create_directory('Recorded-Video')
            self.out = cv2.VideoWriter(f'Recorded-Video/{str(datetime.datetime.now().today()).replace(":", "_")}.mp4',
                                      cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))
            self.b2["text"] = 'STOP 🔴'
        else:
            self.is_recording_started = False
            self.b2["text"] = 'RECORD 🔴'
            if self.out:
                self.out.release()

    # Function to run the main application loop
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # YOLO object detection
            results = self.model(frame, device="mps")  # for MacOS, speeds up YOLO
            self.result = results[0]

            self.bboxes = np.array(self.result.boxes.xyxy.cpu(), dtype="int")
            self.classes = np.array(self.result.boxes.cls.cpu(), dtype="int")
            self.probs = np.array(self.result.boxes.cls.cpu(), dtype="int")

            # Object detection
            self.detected_objects = []

            self.perform_detection(frame)

            if self.is_recording_started and self.out:
                self.out.write(frame)

            # Update Tkinter Image Label
            self.img1 = frame.copy()
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.camera['image'] = img
            self.root.update()

        # Release everything if the job is finished
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        print("The video was successfully saved")

    # Function to update text labels
    def textLabels(self, text1=None, text2=None, text3=None):
        # Clear any existing labels
        for widget in self.text_frame.winfo_children():
            widget.destroy()

        print(text1)
        print(text2)
        print(text3)

        # Create labels based on the provided texts
        label1 = tk.Label(self.text_frame, text=text1, font=self.label_font, fg='#70955B')
        label2 = tk.Label(self.text_frame, text=text2, font=self.label_font, fg='#F2A59C')
        label3 = tk.Label(self.text_frame, text=text3, font=self.label_font, fg='#64AFA8')

        self.text_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.text_frame.grid_rowconfigure(0, weight=1)

        # Grid layout for labels
        label1.grid(row=0, column=0, sticky="nwes")
        label2.grid(row=0, column=1, sticky="nwes")
        label3.grid(row=0, column=2, sticky="nwes")

    # Function to perform object detection
    def perform_detection(self, frame):
        for class_id, bbox, score in zip(self.classes, self.bboxes, self.probs):
            class_name = self.result.names[class_id]
            score = float(score.item())
            (x, y, x2, y2) = bbox

            print(f"class {class_name} conf {score} xywh {x},{y},{x2},{y2}")

            if self.is_detection_started:
                if class_name not in ["fork", "spoon", "knife"]:
                    continue
                else:
                    # make sure that there is only one detection for that class
                    if class_name in self.detected_objects:
                        continue

            # Get details of object
            self.detected_objects.append(class_name)

            # get details of object
            self.detected_objects.append(class_name)
            self.placement[f"{class_name}_x"] = x

            # Draw bounding box and class name
            if self.is_detection_started:
                cv2.rectangle(frame, (x, y), (x2, y2), self.color_mapping[class_name], 2)
                cv2.putText(frame, "{} [{:.2f}]".format(str(class_name), float(x)), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_mapping[class_name], 2)
            else:
                color = tuple(map(int, self.class_colors[class_id]))
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                cv2.putText(frame, "{} [{:.2f}]".format(str(class_name), float(x)), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if self.is_detection_started:

            # Update text labels
            self.textLabels(
                text1="No fork detected" if "fork" not in self.detected_objects else None,
                text2="No knife detected" if "knife" not in self.detected_objects else None,
                text3="No spoon detected" if "spoon" not in self.detected_objects else None
            )

            # Check positions
            if "fork" in self.detected_objects and "knife" in self.detected_objects and "spoon" in self.detected_objects:
                fork_x = self.placement["fork_x"]
                knife_x = self.placement["knife_x"]
                spoon_x = self.placement["spoon_x"]

                print("f: ", fork_x)
                print("k: ", knife_x)
                print("s: ", spoon_x)

                if fork_x < knife_x:
                    if knife_x < spoon_x: self.textLabels(text2="Correct Utensil Order")
                    else: self.textLabels(text2="Knife should be after the fork")
                else: self.textLabels(text2="Fork should be on the left side")


if __name__ == "__main__":
    root = tk.Tk()
    app = UtensilManagementApp(root)
    app.run()
