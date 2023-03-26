python test_ADA.py \
  --device '5' \
  --batch-size 1 \
  --img-size 1024 \
  --data "./configs/gwhd2021.yaml" \
  --cfg "./models/yolov4-p7.yaml" \
  --task "test" \
  --dct_type 'reg' \
  --dct_arch 'resnet18' \
  --conf-thres 0.4 \
  --iou-thres 0.5 \
  --weights './pretrained/regDCT.pt'