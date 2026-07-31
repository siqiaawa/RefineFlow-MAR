import os
import os.path
import argparse
import numpy as np
import torch
# import matplotlib.pyplot as plt
import h5py
from PIL import Image

from .build_gemotry import initialization, imaging_geo

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    # data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    # data = data * 255.0
    data = data * 2. - 1.
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

param = initialization()
ray_trafo, FBPOper = imaging_geo(param)


def test_image_clinical(data_path, imag_idx, inner_dir):
    """
    Load clinical data in the format produced by generate_clinical_data.py.

    Differences from test_image:
    - No ground truth; Xma/Sma stand in for Xgt/Sgt only to initialize torch_radon.
    - Each image has one mask, read from testmask.npy[:, :, imag_idx].
    - Each line in the text file points directly to an H5 data file, such as
      patient_0000/000/0.h5.
    """
    txtdir = os.path.join(data_path, 'test_720geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
    data_file = mat_files[imag_idx].strip()
    # Support two text-file formats:
    #   Standard: patient_0000/000/gt.h5 -> use 0.h5 for clinical data.
    #   Clinical: patient_0000/000/0.h5 -> use the path directly.
    if data_file.endswith('gt.h5'):
        data_file = data_file[:-5] + '0.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    Sma = file['ma_sinogram'][()]
    XLI = file['LI_CT'][()]
    SLI = file['LI_sinogram'][()]
    Tr  = file['metal_trace'][()]
    file.close()
    M512 = test_mask[:, :, imag_idx]
    M = M512
    Xma_n = normalize(Xma, image_get_minmax())
    XLI_n = normalize(XLI, image_get_minmax())
    Sma_n = normalize(Sma, proj_get_minmax())
    SLI_n = normalize(SLI, proj_get_minmax())
    Tr_n = 1 - Tr.astype(np.float32)
    Tr_n = np.expand_dims(np.transpose(np.expand_dims(Tr_n, 2), (2, 0, 1)), 0)
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    # Xgt/Sgt are unavailable; use Xma/Sma only to initialize torch_radon.
    return (torch.Tensor(Xma_n).cuda(), torch.Tensor(XLI_n).cuda(),
            torch.Tensor(Xma_n).cuda(),   # Xgt placeholder
            torch.Tensor(Mask).cuda(),
            torch.Tensor(Sma_n).cuda(), torch.Tensor(SLI_n).cuda(),
            torch.Tensor(Sma_n).cuda(),   # Sgt placeholder
            torch.Tensor(Tr_n).cuda())


def test_image(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'test_720geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir.strip()[:-5]  # remove 'gt.h5'
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir.strip())
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma= file['ma_CT'][()]
    Sma = file['ma_sinogram'][()]
    XLI = file['LI_CT'][()]
    SLI = file['LI_sinogram'][()]
    Tr = file['metal_trace'][()]
    Sgt = np.asarray(ray_trafo(Xgt))
    file.close()
    M512 = test_mask[:,:,mask_idx]
    M = M512  # The 512x512 dental CT data does not need resizing.
    Xma = normalize(Xma, image_get_minmax())  # *255
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    Sma = normalize(Sma, proj_get_minmax())
    Sgt = normalize(Sgt, proj_get_minmax())
    SLI = normalize(SLI, proj_get_minmax())
    Tr = 1 - Tr.astype(np.float32)
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)), 0)  # 1*1*h*w
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(Mask).cuda(), \
       torch.Tensor(Sma).cuda(), torch.Tensor(SLI).cuda(), torch.Tensor(Sgt).cuda(), torch.Tensor(Tr).cuda()
