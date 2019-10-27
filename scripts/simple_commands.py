
"""
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]

Optional arguments:

RESULT_FILE: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
EVAL_METRICS: Items to be evaluated on the results. Allowed values are: proposal_fast, proposal, bbox, segm, keypoints.
--show: If specified, detection results will be ploted on the images and shown in a new window. It is only applicable to single GPU testing. Please make sure that GUI is available in your environment, otherwise you may encounter the error like cannot connect to X server.


# test
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth --show
python demo/webcam_demo.py configs/faster_rcnn_r50_fpn_1x.py checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth



# train
python tools/train.py configs/faster_rcnn_r50_fpn_1x_train.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r50_fpn_1x/2019-10-22_train2
python tools/train.py configs/faster_rcnn_r50_fpn_1x_train.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r50_fpn_1x/2019-10-22_train2 --autoscale-lr
python tools/train.py configs/fp16/retinanet_r50_fpn_fp16_1x_train.py --work_dir /home/moshes2/Projects/mmdetection/results/retinanet_r50_fpn_fp16_1x/2019-10-23_train2 --autoscale-lr
python tools/train.py configs/fp16/faster_rcnn_r34_fpn_fp16_1x.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r34_fpn_fp16_1x/2019-10-23_train --autoscale-lr
python tools/train.py configs/faster_rcnn_r34_fpn_1x.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r34_fpn_1x/2019-10-24_train --autoscale-lr
python tools/train.py configs/faster_rcnn_r50_fpn_1x_train.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r50_fpn_1x/2019-10-24_train --autoscale-lr --resume_from checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth
python tools/train.py configs/fp16/faster_rcnn_r34_fpn_fp16_1x.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r34_fpn_fp16_1x/2019-10-24_train --autoscale-lr
# This is the command used to train the first r34 network
python tools/train.py configs/faster_rcnn_r34_fpn_1x.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r34_fpn_1x/2019-10-24_train --autoscale-lr --resume_from checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth
python tools/train.py configs/faster_rcnn_r34_fpn_1x_grayscale.py --work_dir /home/moshes2/Projects/mmdetection/results/faster_rcnn_r34_fpn_1x_grayscale/2019-10-24_train --autoscale-lr --resume_from checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth
python tools/train.py configs/faster_rcnn_r34_fpn_1x_grayscale.py --work_dir /home/moshes2/Projects/mmdetection/results//faster_rcnn_r34_fpn_1x_grayscale/2019-10-27_train --autoscale-lr --resume_from results/faster_rcnn_r34_fpn_1x/2019-10-24_train/epoch_32.pth



# test
python tools/test.py configs/faster_rcnn_r50_fpn_1x_train.py results/2019-09-15_train/latest.pth --out results/2019-09-15_train/eval_epoch8.pkl --eval proposal bbox
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth --out results/checkpoints/faster_rcnn_r50_c4_1x.pkl --eval proposal bbox
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth --out results/checkpoints/faster_rcnn_r50_c4_2x.pkl --eval proposal bbox

python tools/test.py configs/faster_rcnn_r34_fpn_1x.py results/faster_rcnn_r34_fpn_1x/2019-10-24_train/epoch_32.pth --out results/faster_rcnn_r34_fpn_1x/2019-10-24_train/eval_results.pkl --json_out results/faster_rcnn_r34_fpn_1x/2019-10-24_train/eval_results.json --eval proposal bbox --show


# plot losses
python tools/analyze_logs.py plot_curve results/2019-09-15_train/20190915_162558.log.json --keys loss loss_cls loss_bbox loss_rpn_cls loss_rpn_bbox --legend loss loss_cls loss_bbox loss_rpn_cls loss_rpn_bbox --out losses.pdf
python tools/analyze_logs.py plot_curve results/2019-09-15_train/20190915_162558.log.json --keys loss loss_cls loss_bbox loss_rpn_cls loss_rpn_bbox --legend loss loss_cls loss_bbox loss_rpn_cls loss_rpn_bbox --out losses.pdf


"""