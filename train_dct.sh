CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  --master_port 10003 \
  train_dct.py \
  --batch-size 16 \
  --img 1024 1024 \
  --data "./configs/gwhd2021.yaml" \
  --hyp "./configs/hyp.head.yaml" \
  --cfg "./models/yolov4-p7.yaml" \
  --sync-bn  \
  --epochs 50 \
  --dct_type 'reg' \
  --dct_arch 'resnet18' \
  --weights './runs/gwhd2021_yolov4-p7_baseline/last.pt' \
  --name 'gwhd2021_yolov4-p7_DCT' \
  --head_train