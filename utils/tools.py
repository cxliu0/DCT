import numpy as np

def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def acc_image(boxes_true, boxes_pred, thresholds=[0.5]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, x2, y2)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, x2, y2)
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # if there are no gt boxes
    if len(boxes_true) == 0:
        if len(boxes_pred) > 0:
            return 0.0
        else:
            return 1.0
    
    acc_total = 0
    iou_array = bbox_overlaps(boxes_true.copy(), boxes_pred.copy())  # COPY !!!
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
                
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou_array[i,j]
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        acc = tp / (tp + fn + fp)
        acc_total += acc
    
    return acc_total / len(thresholds)

def encode_boxes(boxes):
    if len(boxes) >0:
        #boxes = [" ".join([str(int(i)) for i in item]) for item in boxes]
        boxes = [" ".join([str(i) for i in item]) for item in boxes]
        BoxesString = ";".join(boxes)
    else:
        BoxesString = "no_box"
    return BoxesString
