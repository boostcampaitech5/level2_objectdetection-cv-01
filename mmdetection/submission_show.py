import cv2
import pandas
import random
from tqdm import tqdm
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

confidence = 0.5

DF = pandas.read_csv('/opt/ml/mmdetection/work_dirs/deformable_detr/submission_latest.csv')

for i in tqdm(range(len(DF))):
    file_name = str(i).zfill(4) + '.jpg'

    image = cv2.imread('/opt/ml/dataset/train/'+file_name)
    D = DF['PredictionString'][i].split()
    for j in range(len(D)//6):
        if float(D[j*6+1])<confidence:
            continue
        a=random.randint(64, 191)
        b=random.randint(64, 191)
        c=random.randint(64, 191)
        x1 = int(float(D[j*6+2]))
        y1 = int(float(D[j*6+3]))
        x2 = int(float(D[j*6+4]))
        y2 = int(float(D[j*6+5]))
        C = int(D[j*6])
        cv2.rectangle(image, (x1, y1), (x2, y2), (a, b, c), 3)
        cv2.putText(image, classes[C], ((x1+x2)//2, y2-10), cv2.FONT_ITALIC, 1, (a, b, c), 3)
    cv2.imwrite('/opt/ml/workspace/imagebycsv/'+file_name, image)