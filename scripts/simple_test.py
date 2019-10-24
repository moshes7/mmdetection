from __future__ import absolute_import
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = '../configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
show_result(img, result, model.CLASSES)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, out_file='result.jpg')

print('Done')