from __future__ import print_function
from os.path import join
import argparse
import torch
import torch.nn as nn
import time
import math
from torch.autograd import Variable
from PIL import Image
import glob
import os
from torchvision.transforms import ToTensor
import numpy as np
'''
python3 run.py --folder /home/hesu215/a-PyTorch-Tutorial-to-Super-Resolution/dataset/testing_lr_images/ --scale_factor 3 --cuda --model model_epoch_80.pth 
'''
# Demonstration settings
parser = argparse.ArgumentParser(description='VDSR PyTorch Demonstration')

parser.add_argument('--folder', type=str, required=True, help="test folder")

parser.add_argument('--model', type=str, required=True,
                    help='model file to use')
parser.add_argument('--scale_factor', type=float,
                    help='factor by which super resolution needed')
parser.add_argument('--cuda', action='store_true', help='use cuda')

parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU ID for using')
opt = parser.parse_args()
opt.gpuids = list(map(int, opt.gpuids))
print(opt)
for f in glob.glob(os.path.join(opt.folder, "*.png")):
    img = Image.open(f).convert('YCbCr')
    y, cb, cr = img.split()

    img = img.resize((int(img.size[0]*opt.scale_factor),
                    int(img.size[1]*opt.scale_factor)), Image.BICUBIC)

    model_name = join("model", opt.model)
    model = torch.load(model_name)
    input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])

    torch.cuda.set_device(opt.gpuids[0])
    with torch.cuda.device(opt.gpuids[0]):
        model = model.cuda()
    model = nn.DataParallel(model, device_ids=opt.gpuids,
                            output_device=opt.gpuids[0])

    if opt.cuda:
        input = input.cuda()

    start_time = time.time()
    out = model(input)
    elapsed_time = time.time() - start_time
    print("===> It takes {:.4f} seconds.".format(elapsed_time))
    out = out.cpu()

    print("type = ", type(out))
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge(
        'YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    output_path = os.path.join("./submission", os.path.basename(f))
    out_img.save(output_path)
    print('output image saved to ', output_path)
