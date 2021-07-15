################################################################################
# Copyright (c) 2019-present, Mahmoud Afifi
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#
# Please, cite the following paper if you use this code:
# Mahmoud Afifi and Michael S. Brown. What else can fool deep learning?
# Addressing color constancy errors on deep neural network performance. ICCV,
# 2019
#
# Email: mafifi@eecs.yorku.ca | m.3afifi@gmail.com
################################################################################


import numpy as np
import numpy.matlib
from PIL import Image
import random as rnd
import os
from os.path import splitext, split, basename, join, exists
import shutil
from datetime import datetime
import pickle
from math import ceil


class WBEmulator:
  def __init__(self):
    # training encoded features
    if os.path.split(os.getcwd())[-1] != 'WB_color_augmenter_python':
      folder = 'WB_color_augmenter_python/'
    else:
      folder = './'
    print(folder)
    self.features = np.load(folder + 'features.npy')
    # mapping functions to emulate WB effects
    self.mappingFuncs = np.load(folder + 'mappingFuncs.npy')
    # weight matrix for histogram encoding
    self.encoderWeights = np.load(folder + 'encoderWeights.npy')
    # bias vector for histogram encoding
    self.encoderBias = np.load(folder + 'encoderBias.npy')
    self.h = 60  # histogram bin width
    self.K = 25  # K value for nearest neighbor searching
    self.sigma = 0.25  # fall off factor for KNN
    # WB & photo finishing styles
    self.wb_photo_finishing = ['_F_AS', '_F_CS', '_S_AS', '_S_CS',
                               '_T_AS', '_T_CS', '_C_AS', '_C_CS',
                               '_D_AS', '_D_CS']

  def encode(self, hist):
    """Generates a compacted feature of a given RGB-uv histogram tensor."""
    histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                (1, int(hist.size / 3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped,
                              [histG_reshaped, histB_reshaped])
    feature = np.dot(hist_reshaped - self.encoderBias.transpose(),
                     self.encoderWeights)
    return feature

  def rgbuv_hist(self, I):
    """Computes an RGB-uv histogram tensor."""
    sz = np.shape(I)  # get size of current image
    if sz[0] * sz[1] > 202500:  # resize if it is larger than 450*450
      factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
      newH = int(np.floor(sz[0] * factor))
      newW = int(np.floor(sz[1] * factor))
      I = imresize(I, output_shape=(newW, newH))
    I_reshaped = I[(I > 0).all(axis=2)]
    eps = 6.4 / self.h
    A = np.arange(-3.2, 3.19, eps)  # dummy vector
    hist = np.zeros((A.size, A.size, 3))  # histogram will be stored here
    Iy = np.linalg.norm(I_reshaped, axis=1)  # intensity vector
    for i in range(3):  # for each histogram layer, do
      r = []  # excluded channels will be stored here
      for j in range(3):  # for each color channel do
        if j != i:  # if current channel does not match current layer,
          r.append(j)  # exclude it
      Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
      Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
      hist[:, :, i], _, _ = np.histogram2d(
        Iu, Iv, bins=self.h, range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2,
        weights=Iy)
      norm_ = hist[:, :, i].sum()
      hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
    return hist

  def generateWbsRGB(self, I, outNum=10):
    """Generates outNum new images of a given image I."""
    assert (outNum <= 10)
    I = to_numpy(I)  # convert to double
    feature = self.encode(self.rgbuv_hist(I))
    if outNum < len(self.wb_photo_finishing):
      wb_pf = rnd.sample(self.wb_photo_finishing, outNum)
      inds = []
      for j in range(outNum):
        inds.append(self.wb_photo_finishing.index(wb_pf[j]))

    else:
      wb_pf = self.wb_photo_finishing
      inds = list(range(0, len(wb_pf)))
    synthWBimages = []

    D_sq = np.einsum('ij, ij ->i', self.features,
                     self.features)[:, None] + np.einsum(
      'ij, ij ->i', feature, feature) - 2 * self.features.dot(feature.T)

    # get smallest K distances
    idH = D_sq.argpartition(self.K, axis=0)[:self.K]
    dH = np.sqrt(
      np.take_along_axis(D_sq, idH, axis=0))
    weightsH = np.exp(-(np.power(dH, 2)) /
                      (2 * np.power(self.sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    for i in range(len(inds)):  # for each of the retried training examples,
      ind = inds[i]  # for each WB & PF style,
      # generate a mapping function
      mf = sum(np.reshape(np.matlib.repmat(weightsH, 1, 27),
                          (self.K, 1, 9, 3)) *
               self.mappingFuncs[(idH - 1) * 10 + ind, :])
      mf = mf.reshape(9, 3, order="F")  # reshape it to be 9 * 3
      synthWBimages.append(changeWB(I, mf))  # apply it!
    return synthWBimages, wb_pf


  def computeMappingFunc(self, I, outNum=10):
    """Generates outNum mapping functions of a given image I."""
    assert (outNum <= 10)
    I = to_numpy(I)  # convert to double
    feature = self.encode(self.rgbuv_hist(I))
    if outNum < len(self.wb_photo_finishing):
      wb_pf = rnd.sample(self.wb_photo_finishing, outNum)
      inds = []
      for j in range(outNum):
        inds.append(self.wb_photo_finishing.index(wb_pf[j]))

    else:
      wb_pf = self.wb_photo_finishing
      inds = list(range(0, len(wb_pf)))
    mfs = []

    D_sq = np.einsum('ij, ij ->i', self.features,
                     self.features)[:, None] + np.einsum(
      'ij, ij ->i', feature, feature) - 2 * self.features.dot(feature.T)

    # get smallest K distances
    idH = D_sq.argpartition(self.K, axis=0)[:self.K]
    dH = np.sqrt(
      np.take_along_axis(D_sq, idH, axis=0))
    weightsH = np.exp(-(np.power(dH, 2)) /
                      (2 * np.power(self.sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    for i in range(len(inds)):  # for each of the retried training examples,
      ind = inds[i]  # for each WB & PF style,
      # generate a mapping function
      mf = sum(np.reshape(np.matlib.repmat(weightsH, 1, 27),
                          (self.K, 1, 9, 3)) *
               self.mappingFuncs[(idH - 1) * 10 + ind, :])
      mfs.append(mf.reshape(9, 3, order="F")) # reshape it to be 9 * 3
    return mfs


  def precompute_mfs(self, filenames, outNum=10, target_dir=None):
    """Store mapping functions for a set of files."""
    assert (outNum <= 10)
    if target_dir is None:
      now = datetime.now()
      target_dir = now.strftime('%m-%d-%Y_%H-%M-%S')
    for file in filenames:
      I = to_numpy(Image.open(file))
      mfs = self.computeMappingFunc(I, outNum=outNum)
      out_filename = basename(splitext(file)[0])
      main_dir = split(file)[0]
      if exists(join(main_dir, target_dir)) == 0:
        os.mkdir(join(main_dir, target_dir))
      with open(join(main_dir, target_dir, out_filename + '_mfs.pickle'),
                'wb') as handle:
        pickle.dump(mfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return target_dir


  def delete_precomputed_mfs(self, filenames, target_dir):
    """Delete stored mapping functions for a set of files."""
    for file in filenames:
      out_filename = basename(splitext(file)[0])
      main_dir = split(file)[0]
      os.remove(join(main_dir, target_dir, out_filename + '_mfs.pickle'))


  def open_with_wb_aug(self, filename, target_dir, target_size=None):
    I = Image.open(filename)
    if target_size is not None:
      I = I.resize((target_size, target_size))
    I = to_numpy(I)
    out_filename = basename(splitext(filename)[0])
    main_dir = split(filename)[0]
    with open(join(main_dir, target_dir, out_filename + '_mfs.pickle'),
              'rb') as handle:
      mfs = pickle.load(handle)
      ind = np.random.randint(len(mfs))
      mf = mfs[ind]
      I = changeWB(I, mf)
      return I


  def single_image_processing(self, in_img, out_dir="../results", outNum=10,
                              write_original=1):
    """Applies the WB emulator to a single image in_img."""
    assert (outNum <= 10)
    print("processing image: " + in_img + "\n")
    filename, file_extension = os.path.splitext(in_img)  # get file parts
    I = Image.open(in_img)  # read the image
    # generate new images with different WB settings
    outImgs, wb_pf = self.generateWbsRGB(I, outNum)
    for i in range(outNum):  # save images
      outImg = outImgs[i]  # get the ith output image
      outImg.save(out_dir + '/' + os.path.basename(filename) +
                  wb_pf[i] + file_extension)  # save it
      if write_original == 1:
        I.save(out_dir + '/' + os.path.basename(filename) +
               '_original' + file_extension)

  def batch_processing(self, in_dir, out_dir="../results", outNum=10,
                       write_original=1):
    """Applies the WB emulator to all images in a given directory in_dir."""
    assert (outNum <= 10)
    imgfiles = []
    valid_images = (".jpg", ".bmp", ".png", ".tga")
    for f in os.listdir(in_dir):
      if f.lower().endswith(valid_images):
        imgfiles.append(os.path.join(in_dir, f))
    for in_img in imgfiles:
      print("processing image: " + in_img + "\n")
      filename, file_extension = os.path.splitext(in_img)
      I = Image.open(in_img)
      outImgs, wb_pf = self.generateWbsRGB(I, outNum)
      for i in range(outNum):  # save images
        outImg = outImgs[i]  # get the ith output image
        outImg.save(out_dir + '/' + os.path.basename(filename) +
                    wb_pf[i] + file_extension)  # save it
        if write_original == 1:
          I.save(out_dir + '/' + os.path.basename(filename) + '_original' +
                 file_extension)

  def trainingGT_processing(self, in_dir, out_dir, gt_dir, out_gt_dir, gt_ext,
                            outNum=10, write_original=1):
    """Applies the WB emulator to all training images in in_dir and
        generates corresponding GT files"""
    imgfiles = []  # image files will be saved here
    gtfiles = []  # ground truth files will be saved here
    # valid image file extensions (modify it if needed)
    valid_images = (".jpg", ".bmp", ".png", ".tga")
    for f in os.listdir(in_dir):  # for each file in in_dir
      if f.lower().endswith(valid_images):
        imgfiles.append(os.path.join(in_dir, f))

    # get corresponding ground truth files
    for in_img in imgfiles:
      filename, file_extension = os.path.splitext(in_img)
      gtfiles.append(os.path.join(gt_dir, os.path.basename(filename) +
                                  gt_ext))

    for in_img, gtfile in zip(imgfiles, gtfiles):
      print("processing image: " + in_img + "\n")
      filename, file_extension = os.path.splitext(in_img)
      gtbasename, gt_extension = os.path.splitext(gtfile)
      gtbasename = os.path.basename(gtbasename)
      I = Image.open(in_img)
      # generate new images with different WB settings
      outImgs, wb_pf = self.generateWbsRGB(I, outNum)
      for i in range(outNum):
        outImg = outImgs[i]
        outImg.save(out_dir + '/' + os.path.basename(filename) + wb_pf[i] +
                    file_extension)  # save it
        shutil.copyfile(gtfile,  # copy corresponding gt file
                        os.path.join(out_gt_dir, gtbasename + wb_pf[i] +
                                     gt_extension))

        if write_original == 1:  # if write_original flag is true
          I.save(out_dir + '/' + os.path.basename(filename) + '_original' +
                 file_extension)
          # copy corresponding gt file
          shutil.copyfile(gtfile, os.path.join(
            out_gt_dir, gtbasename + '_original' + gt_extension))


def changeWB(input, m):
  """Applies a mapping function m to a given input image."""
  sz = np.shape(input)  # get size of input image
  I_reshaped = np.reshape(input, (int(input.size / 3), 3),
                          order="F")
  kernel_out = kernelP9(I_reshaped)  # raise input image to a higher-dim space
  # apply m to the input image after raising it the selected higher degree
  out = np.dot(kernel_out, m)
  out = outOfGamutClipping(out)  # clip out-of-gamut pixels
  # reshape output image back to the original image shape
  out = out.reshape(sz[0], sz[1], sz[2], order="F")
  out = to_image(out)
  return out


def kernelP9(I):
  """Kernel function: kernel(r, g, b) -> (r, g, b, r2, g2, b2, rg, rb, gb)"""
  return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 0],
                        I[:, 1] * I[:, 1], I[:, 2] * I[:, 2], I[:, 0] * I[:, 1],
                        I[:, 0] * I[:, 2], I[:, 1] * I[:, 2])))


def outOfGamutClipping(I):
  """Clips out-of-gamut pixels."""
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I


def to_numpy(im):
  """Returns a double numpy image [0,1] of the uint8 im [0,255]."""
  return np.array(im) / 255


def to_image(im):
  """Returns a PIL image from a given numpy [0-1] image."""
  return Image.fromarray(np.uint8(im * 255))


### Imresize code

# source: https://github.com/fatheral/matlab_imresize

def deriveSizeFromScale(img_shape, scale):
  output_shape = []
  for k in range(2):
    output_shape.append(int(ceil(scale[k] * img_shape[k])))
  return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
  scale = []
  for k in range(2):
    scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
  return scale


def triangle(x):
  x = np.array(x).astype(np.float64)
  lessthanzero = np.logical_and((x >= -1), x < 0)
  greaterthanzero = np.logical_and((x <= 1), x >= 0)
  f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
  return f


def cubic(x):
  x = np.array(x).astype(np.float64)
  absx = np.absolute(x)
  absx2 = np.multiply(absx, absx)
  absx3 = np.multiply(absx2, absx)
  f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
    -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2))
  return f


def contributions(in_length, out_length, scale, kernel, k_width):
  if scale < 1:
    h = lambda x: scale * kernel(scale * x)
    kernel_width = 1.0 * k_width / scale
  else:
    h = kernel
    kernel_width = k_width
  x = np.arange(1, out_length + 1).astype(np.float64)
  u = x / scale + 0.5 * (1 - 1 / scale)
  left = np.floor(u - kernel_width / 2)
  P = int(ceil(kernel_width)) + 2
  ind = np.expand_dims(left, axis=1) + np.arange(
    P) - 1  # -1 because indexing from 0
  indices = ind.astype(np.int32)
  weights = h(
    np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
  weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
  aux = np.concatenate(
    (np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(
    np.int32)
  indices = aux[np.mod(indices, aux.size)]
  ind2store = np.nonzero(np.any(weights, axis=0))
  weights = weights[:, ind2store]
  indices = indices[:, ind2store]
  return weights, indices


def imresizemex(inimg, weights, indices, dim):
  in_shape = inimg.shape
  w_shape = weights.shape
  out_shape = list(in_shape)
  out_shape[dim] = w_shape[0]
  outimg = np.zeros(out_shape)
  if dim == 0:
    for i_img in range(in_shape[1]):
      for i_w in range(w_shape[0]):
        w = weights[i_w, :]
        ind = indices[i_w, :]
        im_slice = inimg[ind, i_img].astype(np.float64)
        outimg[i_w, i_img] = np.sum(
          np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
  elif dim == 1:
    for i_img in range(in_shape[0]):
      for i_w in range(w_shape[0]):
        w = weights[i_w, :]
        ind = indices[i_w, :]
        im_slice = inimg[i_img, ind].astype(np.float64)
        outimg[i_img, i_w] = np.sum(
          np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
  if inimg.dtype == np.uint8:
    outimg = np.clip(outimg, 0, 255)
    return np.around(outimg).astype(np.uint8)
  else:
    return outimg


def imresizevec(inimg, weights, indices, dim):
  wshape = weights.shape
  if dim == 0:
    weights = weights.reshape((wshape[0], wshape[2], 1, 1))
    outimg = np.sum(
      weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
  elif dim == 1:
    weights = weights.reshape((1, wshape[0], wshape[2], 1))
    outimg = np.sum(
      weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)),
      axis=2)
  if inimg.dtype == np.uint8:
    outimg = np.clip(outimg, 0, 255)
    return np.around(outimg).astype(np.uint8)
  else:
    return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
  if mode == "org":
    out = imresizemex(A, weights, indices, dim)
  else:
    out = imresizevec(A, weights, indices, dim)
  return out


def imresize(I, scalar_scale=None, output_shape=None, mode="vec"):
  kernel = cubic

  kernel_width = 4.0
  # Fill scale and output_size
  if scalar_scale is not None:
    scalar_scale = float(scalar_scale)
    scale = [scalar_scale, scalar_scale]
    output_size = deriveSizeFromScale(I.shape, scale)
  elif output_shape is not None:
    scale = deriveScaleFromSize(I.shape, output_shape)
    output_size = list(output_shape)
  else:
    print('Error: scalar_scale OR output_shape should be defined!')
    return
  scale_np = np.array(scale)
  order = np.argsort(scale_np)
  weights = []
  indices = []
  for k in range(2):
    w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel,
                           kernel_width)
    weights.append(w)
    indices.append(ind)
  B = np.copy(I)
  flag2D = False
  if B.ndim == 2:
    B = np.expand_dims(B, axis=2)
    flag2D = True
  for k in range(2):
    dim = order[k]
    B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
  if flag2D:
    B = np.squeeze(B, axis=2)
  return B


def convertDouble2Byte(I):
  B = np.clip(I, 0.0, 1.0)
  B = 255 * B
  return np.around(B).astype(np.uint8)

