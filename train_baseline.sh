CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  --master_port 10002 \
  train_baseline.py \
  --batch-size 16 \
  --img 1024 1024 \
  --data "./configs/gwhd2021.yaml" \
  --hyp "./configs/hyp.finetune.yaml" \
  --cfg "./models/yolov4-p7.yaml" \
  --sync-bn  \
  --epochs 300 \
  --weights "./pretrained/yolov4-p7.pt" \
  --name 'gwhd2021_yolov4-p7_baseline'
  
  