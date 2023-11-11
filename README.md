# object-pose-detector
This is an educational project serving as a bachelor's thesis on the topic of 3D object pose measurement., comprising of three main sections:
- Filtering the annotation JSON file of the COCO dataset for attaining a custom dataset of vital objects only
- Training the YOLOv5 model on the obtained dataset with fine-tuning parameters for optimized performance and testing on real cases of images captured from the Intel RealSense depth camera.
- Developing a module that feeds the RGB image of the camera into the detection model. The predicted bounding boxes are then aligned with respect to the depth frame for computing the 3D pose of the detected objects.
