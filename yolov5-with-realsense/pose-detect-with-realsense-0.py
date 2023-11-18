import cv2
import numpy as np
import time
import torch
import yaml
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.plots import colors
from utils.torch_utils import select_device

try:
    import pyrealsense2 as rs
    has_realsense = True
except ImportError:
    has_realsense = False

class ReaslSense_Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.align = rs.align(rs.stream.color)  # Align incoming depth frames to color frames
        #pipeline.start(config)
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

    def get_info(self, depth_frame):
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        ppx = depth_intrin.ppx
        ppy = depth_intrin.ppy
        fx = depth_intrin.fx
        fy = depth_intrin.fy
        return ppx, ppy, fx, fy

    #------------------------------------------------------------
    def get_object_3d_coordinates(self, depth_frame, u, v):
        # Obtain distance value in centimeters
        depth = depth_frame.get_distance(u, v)
        distance_in_cm = depth * 100

        # Get intrinsics parameters of camera
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        # Deproject pixel coordinates (u,v) and depth value to 3D coordinates (X,Y,Z) in cm
        X_3d = (u - depth_intrin.ppx) * distance_in_cm / depth_intrin.fx
        Y_3d = (v - depth_intrin.ppy) * distance_in_cm / depth_intrin.fy
        Z_3d = distance_in_cm
        return X_3d, Y_3d, Z_3d

def box_label(im, depth_im, box, label, line_width, color=(128, 128, 128), txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    lw = line_width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    x, y = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(im,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)
    cv2.rectangle(depth_im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 1)
    cv2.circle(depth_im, (x, y), 3, (255, 255, 255), -1)

def color_detect(color_image, box):
    color_ranges = {
        'red': ([0, 5, 0], [7, 255, 255]),
        'orange': ([8, 5, 0], [19, 255, 255]),
        'yellow': ([20, 5, 0], [34, 255, 255]),
        'green': ([35, 5, 0], [81, 255, 255]),
        'blue': ([82, 5, 0], [128, 255, 255]),
        'violet': ([129, 5, 0], [147, 255, 255]),
        'pink': ([148, 5, 0], [179, 255, 255]),
    }
    
    x1, y1, x2, y2 = box
    bgr_box = color_image[int(y1):int(y2), int(x1):int(x2)]
    hsv_box = cv2.cvtColor(bgr_box, cv2.COLOR_BGR2HSV)

    dominant_color = None
    max_contour_area = 0
    # dominant_contour = None

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv_box, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)

            # Define a threshold for considering it as an object
            min_object_area = 0.7*(x2-x1)*(y2-y1)  # Adjust this threshold as needed

            if contour_area > min_object_area:
                if contour_area > max_contour_area:
                    max_contour_area = contour_area
                    dominant_color = color
                    # dominant_contour = contour

    # Draw only the dominant contour on the original color image
    # if dominant_contour is not None:
    #     final_contour = dominant_contour + (int(box[0]), int(box[1]))
    #     cv2.drawContours(color_image, [final_contour], 0, (0, 255, 0), 2) 
    return dominant_color # color_image, dominant_color

def detect_with_realsense(pipeline):
    start_time = time.time()
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_frames = camera.align.process(frames)

    # Get aligned frames
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # convert color image and depth image to numpy array format for image processing
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    # Preprocess the color image to feed into the model    
    img0 = letterbox(color_image, new_shape=480)[0]     # scale the color image to a new shape
    img = img0[:, :, ::-1].transpose(2, 0, 1)    # BGR to RGB and HWC to CHW format
    img = np.ascontiguousarray(img)     # array contiguous in memory (numpy requirement)
    img = torch.from_numpy(img).to(device)      # convert numpy array to torch tensor
    img = img.float()  # uint8 to fp32
    img /= 255.0    # scale the pixel values from 0-255 to 0-1
    if img.ndimension() == 3:       
        img = img.unsqueeze(0)      # add a dimension at index 0 since the model expects a batch of images
    
    # Inference
    pred = model(img, augment=False)[0]

    conf_thres = app.confidence_slider.get() / 100.0  # Scale the slider value to the desired range
    iou_thres = app.iou_slider.get() / 100.0  # Scale the slider value to the desired range
    app.confidence_var.set(f'{conf_thres:.2f}')
    app.iou_var.set(f'{iou_thres:.2f}')

    # Apply NMS to get the detections with the highest confidence score and IoU threshold 
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

    # Process detections because they are in a different format for each batch (batch size > 1) and different from the output of the non_max_suppression function (batch size = 1) 
    det = pred[0]

    det_num = [0] * len(names)
    if len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], color_image.shape).round()      # Rescale bounding boxes from 320 to 640 to fit the original color image size 
        for cls in det[:, -1].unique():
            c = int(cls) 
            n = (det[:, -1] == cls).sum()  # total detections per class
            det_num[c] = n.item()
        
        for idx, (*box, conf, cls) in enumerate(reversed(det)):
            c = int(cls)  # float to integer class number
            label = f'{names[c]} {conf:.2f}'
            box_label(color_image, depth_colormap, box, label, line_width=2, color=colors(c, True))

            # Get center coordinates of the object
            x_c = int((box[0] + box[2]) / 2)
            y_c = int((box[1] + box[3]) / 2)

            # Get 3D coordinates of the object
            x_3d, y_3d, z_3d = camera.get_object_3d_coordinates(depth_frame, x_c, y_c)
            
            # Get the dominant color of the object
            if names[c] == 'cup':
                dominant_color = color_detect(color_image, box)
            else:
                dominant_color = None

            # Process user input to show object 3D pose
            class_input = app.class_entry.get()

            if (class_input == names[c] and det_num[c] == 1) or (class_input == f'{dominant_color}' + names[c]):
                app.X_var.set(f'{x_3d:.2f}')
                app.Y_var.set(f'{y_3d:.2f}')
                app.Z_var.set(f'{z_3d:.2f}')

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time) if end_time != start_time else 0.000001
    start_time = time.time()
    cv2.putText(color_image, f'FPS: {fps:.0f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize the frames to fit the GUI window
    resized_color_image = cv2.resize(color_image, interpolation=cv2.INTER_AREA, dsize=(480,360))
    resized_depth_colormap = cv2.resize(depth_colormap, interpolation=cv2.INTER_AREA, dsize=(480,360))

    # Convert the frames from BGR to RGB (PIL uses RGB) for displaying in Tkinter Canvas 
    resized_color_image = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2RGB)
    resized_depth_colormap = cv2.cvtColor(resized_depth_colormap, cv2.COLOR_BGR2RGB)

    # Convert the frames to ImageTk format 
    color_img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized_color_image))
    depth_img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized_depth_colormap))

    #---------Update the GUI window-------------
    # Update video frame
    app.color_label.config(image=color_img_tk)
    app.color_label.image = color_img_tk

    app.depth_label.config(image=depth_img_tk)
    app.depth_label.image = depth_img_tk

    # Update detection number on text area
    sum_text = f'Class - #det \n\n' 
    for i in range(len(names)):
        sum_text += f'{names[i]}: {det_num[i]} \n'
    app.text_area.delete('1.0', tk.END)
    app.text_area.insert(tk.END, sum_text)
    app.text_area.tag_configure("bold", font=("Helvetica", 11, "bold"))
    app.text_area.tag_add("bold", "1.0", "1.end")

    # Schedule the next frame update after 1 millisecond
    root.after(1,detect_with_realsense, pipeline)


class RealSense_App:
    def __init__(self, root):
        self.root = root
        self.root.title('RealSense Object Detection and 3D Pose Estimation')
        self.create_gui()  # Call create_gui to set up the GUI elements

    def clear_pose(self):
        # self.show_pose_var = False
        self.class_entry.delete(0, 'end')
        self.X_var.set('')
        self.Y_var.set('')
        self.Z_var.set('')    

    def create_gui(self):
        #------------------------------------------Frame 1--------------------------------------------------------#
        # Image display frame
        video = tk.Frame(self.root)
        video.pack(fill='x', pady=1, expand=True)

        video_frame = tk.Frame(video)
        video_frame.grid(row=0, column=0, pady=5)

        frame = tk.LabelFrame(video_frame, text='Video Output', font=('Times New Roman', 11, 'bold'), fg='gold', bg='gray20', bd=5, relief='groove')
        frame.grid(row=0, column=0, padx=5, pady=5)

        frame_color = tk.LabelFrame(frame, text='RGB image', font=('Times New Roman', 9, 'bold'), fg='white', bg='gray20', bd=0)
        frame_color.grid(row=0, column=0, padx=5, pady=5)

        self.color_label = tk.Label(frame_color)
        self.color_label.pack()

        depth_frame = tk.LabelFrame(frame, text='Depth image', font=('Times New Roman', 9, 'bold'), fg='white', bg='gray20', bd=0)
        depth_frame.grid(row=0, column=1, padx=5, pady=5)

        self.depth_label = tk.Label(depth_frame)
        self.depth_label.pack()

        # Text area frame for detection info summary
        det_info_frame = tk.Frame(video_frame, bd=3, relief='groove')
        det_info_frame.grid(row=0, column=1, padx=5, pady=5)

        pose_info = tk.Label(det_info_frame, text='Detect number', font=('Times New Roman', 12, 'bold'), fg='gold', bg='gray20', bd=3, relief='groove')
        pose_info.pack(fill='x', expand=True)

        scrollbar = tk.Scrollbar(det_info_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        
        self.text_area = tk.Text(det_info_frame, height= 24, width =14, yscrollcommand=scrollbar.set)
        self.text_area.pack()
        scrollbar.config(command=self.text_area.yview)

        #-----------------------------------------------------Frame 2------------------------------------------------------#
        info = tk.Frame(self.root)
        info.pack(fill='x', expand=True, pady=0)
        #---------------------------------Object detection Frame---------------------------------------#
        detect_info = tk.LabelFrame(info, text='Object Detection', font=('Times New Roman', 12, 'bold'), fg='gold', bg='gray20', bd=5, relief='groove')
        detect_info.grid(row=0, column=0, padx=5, pady=2)

        style = ttk.Style()
        style.configure('myStyle.Horizontal.TScale', font=('Times New Roman', 8, 'bold'), foreground='gold', background='gray20', bd=3, relief='groove')
        
        self.confidence_label = tk.Label(detect_info, text='Confidence:', font=('Times New Roman', 10, 'bold'), fg='white', bg='gray20')
        self.confidence_label.grid(row=0, column=3, padx=24, pady=4, sticky='w')
        self.confidence_var = tk.StringVar()
        self.confidence_var_label = tk.Label(detect_info, textvariable=self.confidence_var, font=('Times New Roman', 10, 'bold'), fg='black', bg='white', width=5, relief='groove', bd=2)
        self.confidence_var_label.grid(row=0, column=4, padx=5, pady=4, sticky='w')
        self.confidence_slider = ttk.Scale(detect_info, from_=0, to=100, name='confidence', orient=tk.HORIZONTAL, length=220, value=60, style='myStyle.Horizontal.TScale')
        self.confidence_slider.grid(row=0, column=5, padx=5, pady=4, sticky='w')

        self.iou_label = tk.Label(detect_info, text='IoU:', font=('Times New Roman', 9, 'bold'), fg='white', bg='gray20')
        self.iou_label.grid(row=1, column=3, padx=24, pady=4, sticky='w')
        self.iou_var = tk.StringVar()
        self.iou_var_label = tk.Label(detect_info, textvariable=self.iou_var, font=('Times New Roman', 9, 'bold'), fg='black', bg='white', width=5, relief='groove', bd=2)
        self.iou_var_label.grid(row=1, column=4, padx=5, pady=4, sticky='w')
        self.iou_slider = ttk.Scale(detect_info, from_=0, to=100, name='iou', orient=tk.HORIZONTAL, length=220, value=60, style='myStyle.Horizontal.TScale')
        self.iou_slider.grid(row=1, column=5, padx=5, pady=4, sticky='w')

        #-----------------------------------Object pose Frame-------------------------------------------------
        object_pose_frame = tk.LabelFrame(info, text='Object Pose', font=('Times New Roman', 12, 'bold'), fg='gold', bg='gray20', bd=5, relief='groove')
        object_pose_frame.grid(row=0, column=2, padx=5, pady=0)

        object_pose = tk.Frame(object_pose_frame, bg='gray20')
        object_pose.pack()
        # User input frame
        user_input = tk.Frame(object_pose, bg='gray20')
        user_input.grid(row=0, column=0, padx=2, pady=0)

        class_label = tk.Label(user_input, text='Enter class: ', font=('Times New Roman', 10, 'bold'), fg='white', bg='gray20')
        class_label.grid(row=0, column=0, padx=5, pady=4, sticky='w')
        
        self.class_entry = tk.Entry(user_input, font=('Times New Roman', 9, 'bold'), width=18, bd = 3)
        self.class_entry.grid(row=0, column=1, padx=10, pady=4, sticky='w')

        # Pose display frame
        pose = tk.Frame(object_pose, bg='gray20')
        pose.grid(row=0, column=1, padx=5, pady=1)

        pose_label = tk.Label(pose, text='Pose (in cm): ', font=('Times New Roman', 10, 'bold'), fg='white', bg='gray20')
        pose_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')

        X_label = tk.Label(pose, text='X = ', font=('Times New Roman', 10, 'bold'), fg='white', bg='gray20')
        X_label.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.X_var = tk.StringVar()
        X_var_label = tk.Label(pose, textvariable=self.X_var, font=('Times New Roman', 10, 'bold'), fg='black', bg='white', width=10, relief='groove', bd=3)
        X_var_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        Y_label = tk.Label(pose, text='Y = ', font=('Times New Roman', 10, 'bold'), fg='white', bg='gray20')
        Y_label.grid(row=0, column=3, padx=7, pady=5, sticky='w')

        self.Y_var = tk.StringVar()
        Y_var_label = tk.Label(pose, textvariable=self.Y_var, font=('Times New Roman', 10, 'bold'), fg='black', bg='white', width=10, relief='groove', bd=3)
        Y_var_label.grid(row=0, column=4, padx=5, pady=5, sticky='w')

        Z_label = tk.Label(pose, text='Z = ', font=('Times New Roman', 10, 'bold'), fg='white', bg='gray20')
        Z_label.grid(row=0, column=5, padx=7, pady=5, sticky='w')

        self.Z_var = tk.StringVar()
        Z_var_label = tk.Label(pose, textvariable=self.Z_var, font=('Times New Roman', 10, 'bold'), fg='black', bg='white', width=10, relief='groove', bd=3)
        Z_var_label.grid(row=0, column=6, padx=3, pady=5, sticky='w')

        # Clear data button frame
        button_frame = tk.Frame(object_pose_frame, bg='gray20')
        # button_frame.grid(row=1, column=0, padx=5, pady=3)
        button_frame.pack(padx=0, pady=4)

        clear_button = tk.Button(button_frame, text="Clear", font=('Times New Roman', 9, 'bold'), height=1, width=7, relief='groove', bd=2, command=self.clear_pose)
        clear_button.pack(side=tk.LEFT, padx=20)
        #------------------------------------------------------------
        button = tk.Button(self.root, text="Exit", font=('Times New Roman', 10, 'bold'), fg='gold', bg='gray20', width=4, relief='groove', bd=3, command=root.quit) 
        button.pack(side=tk.BOTTOM, padx=5, pady=6, fill='x')

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Read trained weight.pt and data.yaml file to be passed to the model:")
        
    parser.add_argument("--weights", help="path to trained weight file")
    parser.add_argument("--data", help="path to the data configuration file")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")

    args = parser.parse_args()

    device = select_device(args.device)
    model = attempt_load(args.weights, device=device)

    # Load the data.yaml file to get class names
    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)

    names = data['names']

    if has_realsense:
        camera = ReaslSense_Camera()
        pipeline = camera.pipeline

        root = tk.Tk()
        app = RealSense_App(root)
        detect_with_realsense(pipeline)
        app.run()
       
    else:
        print("Intel RealSense library not found. Make sure it is installed.")


# Run command:
    # python pose-detect-with-realsense-0.py --weights "runs/train/thesis-exp/weights/best.pt" --data "runs/train/thesis-exp/data.yaml" --device 0