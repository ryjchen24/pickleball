from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.predict('input_videos/match_01/rally_photos/rally_018_1.jpg', save=True)
