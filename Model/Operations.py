import cv2
import torch
import requests
import numpy as np
import pyttsx3 
from models.experimental import attempt_load  
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


yolov7_dir = 'D:/Model/yolov7'
weights_path = yolov7_dir + '/yolov7.pt'
device = select_device('gpu') 
model = attempt_load(weights_path, map_location=device)
model.to(device).eval()

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

reference_data = {
    'chair': {'width_cm': 49, 'distance_m': 2, 'reference_pixels': 98},
    'bottle': {'width_cm': 8, 'distance_m': 2, 'reference_pixels': 19},
    'tv': {'width_cm': 124, 'distance_m': 2, 'reference_pixels': 195},
    'bench': {'width_cm': 40, 'distance_m': 2, 'reference_pixels': 225},
}

#ESP32 PORT
url = 'http://192.168.160.167/cam-lo.jpg'
input_size = 320
small_window_size = (480, 360)

############################################DISTANCE CALUCTAIONS#########################################################################################
def calculate_focal_length(reference_width_px, actual_width_cm, distance_m, original_frame_width, input_size):
    scaling_factor = original_frame_width / input_size
    adjusted_reference_width_px = reference_width_px * scaling_factor
    return (adjusted_reference_width_px * distance_m * 100) / actual_width_cm




def estimate_distance(focal_length, actual_width_cm, perceived_width_px, original_frame_width, input_size):
    scaling_factor = original_frame_width / input_size
    adjusted_perceived_width_px = perceived_width_px * scaling_factor
    return (actual_width_cm * focal_length) / adjusted_perceived_width_px / 100  # Convert to meters
  #########################################################################################################################################################


#####################################################################ESP32 STREAMING##################################################
detected_objects_array = np.array([])
while True:

    response = requests.get(url)
    if response.status_code == 200:
        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        print("Error: Could not retrieve image from ESP32.")
        break
      
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    original_frame_height, original_frame_width = frame.shape[:2]
    frame_resized = cv2.resize(frame, (input_size, input_size))

##########PREPROCESSING############################################################################
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(frame_rgb).to(device)
    img = img.permute(2, 0, 1).float()
    img = img / 255.0
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
######################################################################################################


  ##########################################PREDICTION AND BOUNDING BOX CREATION##########################################################
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.5, 0.45)
    current_objects = []  
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                class_name = model.names[int(cls)].lower() 
                label = f'{class_name} {conf:.2f}'
                current_objects.append(class_name)
                perceived_width_px = x2 - x1
                print(f"Detected {class_name} with bounding box width: {perceived_width_px} pixels")

              
                if class_name in reference_data and reference_data[class_name]['reference_pixels']:
                    reference_width_px = reference_data[class_name]['reference_pixels']
                    actual_width_cm = reference_data[class_name]['width_cm']
                    distance_m = reference_data[class_name]['distance_m']

                    focal_length = calculate_focal_length(reference_width_px, actual_width_cm, distance_m, original_frame_width, input_size)

                    distance_estimated = estimate_distance(focal_length, actual_width_cm, perceived_width_px, original_frame_width, input_size)
                    
                    if class_name not in detected_objects_array and distance_estimated <= 3: 

                        detected_objects_array = np.append(detected_objects_array, class_name)
                        engine.say(f"Detected {class_name} at {distance_estimated:.2f} meters")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for obj in detected_objects_array:
        if obj not in current_objects:
            detected_objects_array = detected_objects_array[detected_objects_array != obj]

    if not engine._inLoop:
        engine.runAndWait()

    cv2.imshow('YOLOv7 Detection', cv2.resize(frame, small_window_size))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                                                                                                             
cv2.destroyAllWindows()
