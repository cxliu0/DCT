import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
import pandas as pd

from models.yolo import Model
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)
from utils.torch_utils import select_device, time_synchronized
from utils.tools import *

def test(data,
         weights=None,
         batch_size=16,
         imgsz=1024,
         conf_thres=0.4,
         iou_thres=0.5,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False,
         dct_type='reg',
         dct_arch='resnet18',
         task='val',
         csvfile=None,
         cfg="./models/yolov4-p7.yaml",
         ):
    
    # Obtain domain information
    image_names = []
    domains = []
    csv_df = pd.read_csv(csvfile)
    for index, row in csv_df.iterrows():
        image_name, _, domain = row['image_name'], row['BoxesString'], row['domain']
        image_names.append(image_name)
        domains.append(domain)
    domain_cat = list(np.unique(domains))
    domain_num = len(domain_cat)
    accs = [0] * domain_num
    domain_count = pd.value_counts(domains)

    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        # Initialize/load model and set device
        device = select_device(opt.device, batch_size=batch_size)
        merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
        if save_txt:
            out = Path('inference/output')
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)
        
        ckpt = torch.load(weights, map_location=device)
        model = Model(cfg, ch=3, nc=1, dct_type=dct_type, dct_arch=dct_arch).to(device)  # create
        model.load_state_dict(ckpt)
        del ckpt

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                        hyp=None, augment=False, cache=False, pad=0.0, rect=True)[0]

    seen = 0
    names = 'wheat'
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'ADA')
    print(s)
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    results = []
    pbar, nb = enumerate(dataloader), len(dataloader)
    pbar = tqdm(pbar, total=nb)  # progress bar
    for batch_i, (img, targets, paths, shapes) in pbar:

        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0 # Input: 0 - 255 to 0.0 - 1.0

        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img)
            t0 += time_synchronized() - t

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)#inf_out:x1,y1,x2,y2,conf,cls
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            tbox = xywh2xyxy(labels[:, 1:5]) * whwh
            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))

                acc = acc_image(tbox.cpu().numpy(), np.array([]))
                img_name = paths[si].split('/')[-1]
                domain = domains[image_names.index(img_name)]
                accs[domain_cat.index(domain)] += acc / domain_count[domain]

                PredString = "no_box"
            else:
                # Append to text file
                if save_txt:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    txt_path = str(out / Path(paths[si]).stem)
                    pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])  # to original
                    for *xyxy, conf, cls in pred:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Append to pycocotools JSON dictionary
                if save_json:
                    image_id = Path(paths[si]).stem
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                    'category_id': coco91class[int(p[5])],
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[4], 5)})

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                acc = acc_image(tbox.cpu().numpy(), pred[:, :4].cpu().numpy())
                img_name = paths[si].split('/')[-1]
                domain = domains[image_names.index(img_name)]
                accs[domain_cat.index(domain)] += acc / domain_count[domain]
                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

                PredString = encode_boxes(pred[:, :4].cpu().numpy())#det x1,y1,x2,y2,conf,cls

            results.append([paths[si].split('/')[-1],PredString,acc])

        # Plot images
        if batch_i < 1:
            f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            plot_images(img, targets, paths, str(f), names)  # ground truth
            f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
            plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions
    
    ADA = np.mean(accs)
    print(f'ADA:{ADA}')
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 7  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, ADA))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, ADA), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='yolov4-p7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='configs/gwhd2021.yaml', help='*.data path')
    parser.add_argument('--cfg', type=str, default='models/yolov4-p7.yaml', help='model.yaml path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--dct_type', type=str, default='reg')
    parser.add_argument('--dct_arch', type=str, default='resnet18')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()

    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             dct_type=opt.dct_type,
             dct_arch=opt.dct_arch,
             task=opt.task,
             csvfile="./data/gwhd_2021/competition_" + opt.task + '.csv',
             )

