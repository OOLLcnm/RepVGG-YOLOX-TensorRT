# /home/dyp/anaconda3/envs/yolox/bin/python tools/demo.py image -n yolox-s \
# -c /home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/best_ckpt_RepVGG.pth \
# --path /home/dyp/common/dyp/tensorRT_cpp/workspace/infer/kl_1.jpg --conf 0.05 --nms 0.1 --tsize 640 --save_result

# /home/dyp/anaconda3/envs/yolox/bin/python tools/demo.py video -n yolox-s \
# -c /home/dyp/common/dyp/YOLOX/weights/yolox_s.pth --path test3.mp4 --conf 0.5 --nms 0.1 --tsize 640 \
#  --save_result --device gpu


# export2onnx
# /home/dyp/anaconda3/envs/yolox/bin/python tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c weights/yolox_s.pth

# export2TensorRT
# /home/dyp/anaconda3/envs/yolox/bin/python tools/trt.py -n yolox-s -c /home/dyp/common/dyp/YOLOX/weights/yolox_s.pth \


# onnx2TensorRT
# /home/dyp/下载/TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0/TensorRT-7.2.1.6/bin/trtexec --onnx=yolox_s.onnx --saveEngine=yolox_s.trt  --explicitBatch

# TensorRT Demo
# /home/dyp/anaconda3/envs/yolox/bin/python tools/demo.py \
# image -n yolox-s --trt --path assets/dog.jpg \
# --conf 0.05 --nms 0.1 --tsize 640 --save_result \
# -c /home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/model_trt.pth


# backup
# ./yolox yolox_s.engine  -i dog.jpg 
# ./yolox yolox_s.engine  -i test.mp4 
# trtexec --fp16 --buildOnly --onnx=yolox_s.onnx --saveEngine=yolox_s.engine

python tools/train.py -n yolox-s -b 16 --fp16

# python tools/export_onnx.py \
# -c /home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth \
#  -f exps/default/yolox_s.py --output-name=yolox_s.onnx --no-onnxsim

# RepYOLOX.pth to ONNX
# python tools/export_onnx.py \
# -c /home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/best_ckpt_RepVGG.pth \
#  -f exps/default/yolox_s.py --output-name=yolox_s_RepVGG.onnx --no-onnxsim \
#  -d True

# python tools/RepVGG_train2inference.py  \
# /home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth \
# /home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/best_ckpt_RepVGG.pth
