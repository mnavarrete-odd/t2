from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str, default='best.pt')
args = parser.parse_args()

weight = args.weight
model = YOLO(weight, task='detect')
model.export(format='engine', device=0, task='detect', imgsz=[1088, 1920], half=True)