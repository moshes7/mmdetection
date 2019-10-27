
import mmcv
from mmdet.core import coco_eval, results2json
from mmdet.datasets import build_dataset
import os

os.chdir('..') # change dir to mmdetection root

# eval_types = ['bbox']
# eval_types = ['proposal']
eval_types = ['proposal', 'bbox']

# config = 'configs/faster_rcnn_r50_fpn_1x_train.py'
# result_files = 'checkpoints'
# result_files = 'results/2019-09-15_train/eval_epoch2.pkl.bbox.json'
# result_files = 'results/2019-09-15_train/eval_epoch8.pkl.bbox.json'

config = 'configs/faster_rcnn_r34_fpn_1x.py'
result_files = 'results/faster_rcnn_r34_fpn_1x/2019-10-24_train/eval_results.bbox.json'
# outputs = mmcv.load(result_files)[0]

cfg = mmcv.Config.fromfile(config)

dataset = build_dataset(cfg.data.test)

# result_files = results2json(dataset, outputs, args.out)
coco_eval(result_files, eval_types, dataset.coco)

print('Done!')