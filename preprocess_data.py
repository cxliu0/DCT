import csv
import os

path = "./data/gwhd_2021/"
phases = ['train', 'val', 'test']
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels')
os.makedirs(label_path, exist_ok=True)

W = 1024
H = 1024
for phase in phases:
    recordtxt = open(f'{path}/{phase}.txt','w')
    f = open(f'{path}/competition_{phase}.csv','r')
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        # skip header
        if i==0:
            continue
        image_name, BoxesString, domain = row
        Boxeslist = BoxesString.split(';')
        
        # record image name
        recordtxt.write(img_path + '/' + image_name + '\n')

        # record each image
        labelname = image_name.replace('png','txt')
        labeltxt = open(os.path.join(label_path, labelname),'w')
        message = ''
        for box in Boxeslist:
            if box.strip() != 'no_box':
                x1 = box.split()[0]
                y1 = box.split()[1]
                x2 = box.split()[2]
                y2 = box.split()[3]

                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                if x1 >= W or y1 >= H:
                    continue
                x2 = min(x2, W)
                y2 = min(y2, H)

                category = '0'
                cx = (x1 + x2) / (2 * W)
                cy = (y1 + y2) / (2 * H)
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                message = category + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(w) + ' ' + str(h) + '\n'  
            labeltxt.write(message)
        labeltxt.close()
    f.close()
    recordtxt.close()
    print(f'{i} recorded')
            
            
        