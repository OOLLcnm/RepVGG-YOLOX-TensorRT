# 深度学习 狗都不学

将[YOLOX](https://github.com/MegEngine/YOLOX)中的3✖3Conv和BottleNeck换成了[RepVGG](https://github.com/DingXiaoH/RepVGG)中提出的结构重参数化Conv。


可先将pth转换成ONNX模型再使用
[TensorRT_cpp](https://github.com/shouxieai/tensorRT_cpp)进行GPU端部署，在RTX3060上有800FPS。

## Start
Step1. 安装 YOLOX
```shell
见[YOLOX](https://github.com/MegEngine/YOLOX)
```

Step2. 训练
```shell
根据[YOLOX](https://github.com/MegEngine/YOLOX)配置好数据集路径等

可修改YOLOX/yolox/exp/yolox_base.py来控制模型结构
    # -----------------  deploy config------------------ #
    self.deploy = False
    self.rep = True

rep=False YOLOX
rep=True YOLOX + RepVGG

deploy=False YOLOX + RepVGG
deploy=True YOLOX + RepVGG(结构重参数化Conv)


deploy设为False，rep设为True后
python train.py ...进行训练
```
Step3. 结构重参数化Conv
```shell
python tools/RepVGG_train2inference.py  \
 best_ckpt.pth \
 best_ckpt_RepVGG.pth

就得到了Inference模型
```
Step4. TensorRT部署
```shell
#TODO
```