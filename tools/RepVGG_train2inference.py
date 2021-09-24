import argparse
import os
import sys
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

sys.path.append("../RepVGG/")
from repvgg import get_RepVGG_func_by_name, repvgg_model_convert

from yolox.exp import get_exp


parser = argparse.ArgumentParser(description='RepVGG_YOLOX Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file',
                    default="/home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth",)
parser.add_argument('save', metavar='SAVE', help='path to the weights file',
                    default="/home/dyp/common/dyp/YOLOX/YOLOX_outputs/yolox_s/best_ckpt_RepVGG.pth",)
parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
parser.add_argument(
        "-f",
        "--exp_file",
        default="/home/dyp/common/dyp/YOLOX/exps/default/yolox_s.py",
        type=str,
        help="plz input your expriment description file",
    )


def convert():
    args = parser.parse_args()


    exp = get_exp(args.exp_file, args.name)
    # train_model = exp.get_model(deploy=True)
    train_model = exp.get_model(deploy=False)
    # train_model = model(deploy=False)
    # print("--------------------------------------------")
    # for module in train_model.modules():
    #     if hasattr(module, 'switch_to_deploy'):
    #         module.switch_to_deploy()
    #     # if save_path is not None:
        #     torch.save(model.state_dict(), save_path)
    # repvgg_model_convert(train_model, save_path=args.save)
    # repvgg_model_convert(train_model)

    # Origin Code for read
    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    # if 'plus' in args.arch:
    #     train_model.switch_repvggplus_to_deploy()
    #     torch.save(train_model.state_dict(), args.save)
    # else:
    repvgg_model_convert(train_model, save_path=args.save)


if __name__ == '__main__':
    convert()