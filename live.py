import time

import cv2
from ultralytics import YOLO

import pandas as pd
import numpy as np
import os
import subprocess

import utils

from tqdm import tqdm

path = 'vehicles/veh_2.mp4'

model = YOLO('model/yolov8x.pt')


dict_classes = model.model.names

video = cv2.VideoCapture(0)

class_IDS = [0, 1,2, 3, 5, 7]

height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video.get(cv2.CAP_PROP_FPS)

vehicles_counter_in = dict.fromkeys(class_IDS, 0)
vehicles_counter_out = dict.fromkeys(class_IDS, 0)


center = int((width/2) / 100)
left_area = center-50
right_area = center+50
offset = 8

colors_per_class = {
    1: (227,0,0),
    2: (0,255,0),
    3: (0,0,255),
    5: (255,255,0),
    7: (0,255,255)
}

video_name = 'result.mp4'
output_path = "rep_" + video_name
tmp_output_path = "tmp_" + output_path
VIDEO_CODEC = "mp4v"
output_video = cv2.VideoWriter(tmp_output_path,
                               cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                               fps, (width, height))

right_flag=False
left_flag=False
sum_count=0
start = time.time()
while(True):
    if time.time()-start>50:
        break
    _, frame = video.read()

    y_hat = model.predict(frame, conf=0.7, classes=class_IDS, device='mps', verbose=False)

    conf = y_hat[0].boxes.conf.cpu().numpy()
    classes = y_hat[0].boxes.cls.cpu().numpy()

    frame_positions = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data,
                                   columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

    labels = [dict_classes[i] for i in classes]

    cv2.line(frame, (center, 0), (center, height), (25, 0, 255), 1)

    for ix, row in enumerate(frame_positions.iterrows()):
        xmin, ymin, xmax, ymax, confidence, class_ = row[1].astype('int')
        rgb = colors_per_class.get(class_)
        center_x = int((xmax + xmin) / 2)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), rgb, 2)  # box
        cv2.putText(img=frame, text=labels[ix] + ' - ' + str(np.round(conf[ix], 2)),
                    org=(xmin, ymin - 10), fontScale=1,fontFace=cv2.FONT_HERSHEY_PLAIN, color=rgb,thickness=2)

        if (xmax>(center-offset)) and (xmax<(center+offset)):
            vehicles_counter_in[class_] += 1
            sum_count+=1

        # if (center_x > (center)) and (center_x < (right_area)):
        #     right_flag=True
        # elif center_x>right_flag:
        #     right_flag=False
        # elif (center_x < (center)) and (center_x > (left_area)):
        #     left_flag=True
        # elif center_x<left_area:
        #     left_flag=False
        #
        # if(center_x < (center + offset)) and (center_x > (center-offset)):
        #     if right_flag:
        #         vehicles_counter_in[class_]+=1
        #     elif left_flag:
        #         vehicles_counter_out[class_]+=1
        #     else:
        #         vehicles_counter_in[class_]+=1
        #         vehicles_counter_out[class_]+=1
    cv2.putText(img=frame, text=f'COUNT :{sum_count}',
                org=(30, 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
    output_video.write(frame)

# Releasing the video
output_video.release()

if os.path.exists(output_path):
    os.remove(output_path)

subprocess.run(
    ["ffmpeg",  "-i", tmp_output_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_path])

os.remove(tmp_output_path)


print(vehicles_counter_in)
print(vehicles_counter_out)
