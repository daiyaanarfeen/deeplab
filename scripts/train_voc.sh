CUDA_VISIBLE_DEVICES=2 python train_voc.py --data-dir /data/daiyaanarfeen/VOC2012 --restore-from resnet50_v1s.pth --gpu 2 --learning-rate 0.007 --snapshot-dir coarse --aug True
CUDA_VISIBLE_DEVICES=2 python train_voc.py --data-dir /data/daiyaanarfeen/VOC2012 --restore-from coarse/CS_scenes_30000.pth --gpu 2 --learning-rate 0.001 --snapshot-dir fine
