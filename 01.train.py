from ultralytics import YOLO
import os 

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)


results = './Results'
os.makedirs(results, exist_ok=True)
# Train the model
train_result = model.train(data='Dataset/data.yaml', epochs=100, batch=32, imgsz=640, project=results, name='CupDetection')