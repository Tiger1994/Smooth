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

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--dataset", default="/home/tiger/Graduate/datasets/LapSRN/Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")


def _overlap_crop_forward(model, x, shave=6, min_size=160000, use_curriculum=False):
    n_GPUs = 1
    scale = 4
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if use_curriculum:
                _, sr_batch = model(lr_batch)[-1]
            else:
                _, sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            _overlap_crop_forward(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size,
          (w_size - w + w_half):w_size]

    return output


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def ssim(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

opt = parser.parse_args()
cuda = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model_path = '/home/server-248/Lihu/result/idea4/perceptual/lapsrn_rnd_perceptual'
model_name = 'best87'
model = torch.load(model_path+'/'+model_name + '.pth')["model"]

datasets = ['Set14']

result_path = '/home/server-248/Lihu/result/idea4/image/' + model_name
results = {}

for dataset in datasets:
    path = '/media/server-248/SSD/Lihu/eval_new/'+dataset
    save_path = result_path+'/'+dataset
    if not os.path.exists(save_path):  # 如果路径不存在
        os.makedirs(save_path)
    image_list = glob.glob(path + "/*.*")

    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    avg_elapsed_time = 0.0

    for image_name in image_list:
        print("Processing ", image_name)
        im_gt_y = sio.loadmat(image_name)['im_gt_y']
        im_l_y = sio.loadmat(image_name)['im_l_y']

        im_gt_y = im_gt_y.astype(float)
        im_l_y = im_l_y.astype(float)


        im_input = im_l_y / 255.

        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()
        HR_4x = _overlap_crop_forward(model, im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        HR_4x = HR_4x.cpu()

        im_h_y = HR_4x.data[0].numpy().astype(np.float32)

        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0, :, :]

        psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=opt.scale)
        avg_psnr_predicted += psnr_predicted

        # ssim_predicted = SSIM(im_gt_y, im_h_y, shave_border=opt.scale)
        ssim_predicted = ssim(im_gt_y, im_h_y, shave_border=opt.scale)
        avg_ssim_predicted += ssim_predicted

        im_l_ycbcr = sio.loadmat(image_name)['im_l_ycbcr']
        w = im_h_y.shape[0]
        h = im_h_y.shape[1]
        im_l_ycbcr = cv2.resize(im_l_ycbcr, (h, w), interpolation=cv2.INTER_CUBIC)

        im_h_y = im_h_y/255
        im_l_ycbcr[:, :, 0] = im_h_y
        im_l_ycbcr = im_l_ycbcr.astype(np.float32)
        img = im_l_ycbcr
        img[:, :, [1, 2]] = img[:, :, [2, 1]]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        img[img < 0.0] = 0.0
        img[img > 1.0] = 1.0

        # plt.imshow(img)
        # plt.show()
        name = image_name.split('/')[-1]
        name = name.split('.')[0]+'.png'
        plt.imsave(save_path + '/' + name, img)


    print("Scale=", opt.scale)
    print("Dataset=", dataset)
    print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
    print("SSIM predicted=", avg_ssim_predicted / len(image_list))
    print("It takes average {}s for processing".format(avg_elapsed_time / len(image_list)))

    results[dataset] = [avg_psnr_predicted / len(image_list)]
    results[dataset].append(avg_ssim_predicted / len(image_list))


pd_results = pd.DataFrame(results, index=['PSNR', 'SSIM'], columns=datasets)
pd_results.to_csv(result_path+'/'+'result.csv', sep=',')