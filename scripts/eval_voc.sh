CUDA_VISIBLE_DEVICES=6,7 python -m pdb evaluate_voc.py --data-dir /data/daiyaanarfeen/VOC2012 --restore-from fine/CS_scenes_30000.pth --gpu 6,7 --whole True
