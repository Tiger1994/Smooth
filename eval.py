import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from skimage.measure import compare_ssim


cuda = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model_path = '/home/tiger/Smooth/result'
model_name = 'best_72'
model = torch.load(model_path+'/'+model_name + '.pth')["model"]
model.eval()

datasets_path = '/SSD64/Smooth/test'
datasets = ['test_article']

result_path = '/SSD64/Smooth/result' + model_name
results = {}

for dataset in datasets:
    path = datasets_path + '/' + dataset
    save_path = result_path+'/'+dataset
    if not os.path.exists(save_path):  # 如果路径不存在
        os.makedirs(save_path)
    image_list = glob.glob(path + "/*.*")

    avg_elapsed_time = 0.0

    for image_name in image_list:
        print("Processing ", image_name)
        im_input = plt.imread(image_name)
        im_input = np.transpose(im_input, [2, 0, 1])
        im_input = im_input[np.newaxis, :]

        im_input = torch.from_numpy(im_input).float()/255.

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()
        _, _, gt_o = model(im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        gt_o = gt_o.cpu()

        gt_o = gt_o.data[0].numpy().astype(np.float32)

        gt_o = gt_o * 255.
        gt_o[gt_o < 0] = 0
        gt_o[gt_o > 255.] = 255.
        gt_o = gt_o.astype('uint8')

        gt_o = np.transpose(gt_o, [1, 2, 0])
        name = image_name.split('/')[-1]
        name = name.split('.')[0]+'.png'
        plt.imsave(save_path + '/' + name, gt_o)
