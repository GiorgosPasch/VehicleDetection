import cv2
from ultralytics import YOLO

import pandas as pd
import numpy as np
import os
import subprocess
import librosa

import utils
from moviepy.editor import *

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error


path = 'vehicles/'
audio_path = 'results/audio/'
gtfile = 'values.csv'
values = utils.getRealValues(path+gtfile)

model = YOLO('model/yolov8x.pt')

dict_classes = model.model.names
scale_percent = 50
class_IDS = [2, 3, 5, 7]
real_values = {
    'car': [],
    'truck': [],
    'motorcycle': [],
    'bus': []
}
pred_values = {
    'car': [],
    'truck': [],
    'motorcycle': [],
    'bus': []
}

for idx, row in values.iterrows():
    groundtruth=dict(row[1:])
    filename = row['filename']
    video = cv2.VideoCapture(path+filename)
    audio = AudioFileClip(path+filename)
    audiofile =audio_path+filename[:-4]+'.wav'
    audio.write_audiofile(audiofile)
    y, _ = librosa.load(audiofile)

    if max(y) < 0.1:
        continue

    if os.path.exists(audiofile):
        os.remove(audiofile)

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)

    vehicles_counter = dict.fromkeys(class_IDS, 0)

    line = height-60
    offset = 8

    colors_per_class = {
        2: (0,255,0),
        3: (0,0,255),
        5: (255,255,0),
        7: (0,255,255)
    }

    if scale_percent != 100:
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)

    output_path = "results/res_"+ filename
    tmp_output_path = "results/temp_"+ filename
    VIDEO_CODEC = "mp4v"
    output_video = cv2.VideoWriter(tmp_output_path,
                                   cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                                   fps, (width, height))

    sum_count=0
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):

        _, frame = video.read()

        frame = utils.risize_frame(frame, scale_percent)

        y_hat = model.predict(frame, conf=0.7, classes=class_IDS, device='mps', verbose=False)

        conf = y_hat[0].boxes.conf.cpu().numpy()
        classes = y_hat[0].boxes.cls.cpu().numpy()

        frame_positions = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data,
                                       columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

        labels = [dict_classes[i] for i in classes]

        cv2.line(frame, (0, line), (width, line), (255, 0, 255), 1)

        for ix, row in enumerate(frame_positions.iterrows()):
            xmin, ymin, xmax, ymax, confidence, class_ = row[1].astype('int')
            rgb = colors_per_class.get(class_)
            center_y = int((ymax + ymin) / 2)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), rgb, 2)  # box
            cv2.putText(img=frame, text=labels[ix] + ' - ' + str(np.round(conf[ix], 2)),
                        org=(xmin, ymin - 10), fontScale=1,fontFace=cv2.FONT_HERSHEY_PLAIN, color=rgb,thickness=2)

            if (center_y>(line-offset)) and (center_y<(line+offset)):
                vehicles_counter[class_] += 1
                sum_count+=1

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

    for k in groundtruth.keys():
        real_values[k].append(groundtruth.get(k))

    for k in vehicles_counter.keys():
        if k == 2:
            pred_values['car'].append(vehicles_counter.get(k))
        elif k == 3:
            pred_values['motorcycle'].append(vehicles_counter.get(k))
        elif k == 5:
            pred_values['bus'].append(vehicles_counter.get(k))
        elif k == 7:
            pred_values['truck'].append(vehicles_counter.get(k))


for k in real_values.keys():
    mae = mean_absolute_error(real_values.get(k), pred_values.get(k))
    print(real_values.get(k))
    print(pred_values.get(k))
    print(mae)