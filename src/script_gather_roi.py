#!/usr/bin/env python
# coding: utf-8
import pickle
import random
import numpy as np
from pathlib import Path as P
from configparser import ConfigParser
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import argparse
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter1d
import sys
from tiff import Tiff


STEP = None
MARGIN = None


def main(args):
    img = Tiff(args.file)
    masks = load_masks(args.name, args.dir)
    total_shape = np.asarray(img.img).shape
    red_masks = filter_masks(masks)
    flatmask = flat_depth_mask(masks, total_shape)
    show_segmentation(img, flatmask, args.dir / "segmentation.png")
    onemask = mask_to_one(masks, total_shape)
    show_segmentation_color(img, onemask, args.dir / "segmentation_color.png")
    pixels = extract_pixels(masks, img)
    print(f"Found {len(pixels)} masks to analyze")

    df = get_df_scores(pixels)
    df.to_csv(P(args.dir) / "maskrcnn_rois.csv")
    export_hists(df, args.dir / "histogram_roi")
    export_metrics(df, args.dir / "metrics.txt")
    export_pixels(pixels, args.dir / "pixels_roi.pck")

def export_hists(df, figname):
    for c in df.columns:
        plt.hist(df[c], bins=30)
        plt.savefig(str(figname) + f"_{c}")
        plt.close()

def export_metrics(df, filename):
    f = open(filename, "w")
    for i, row in df.corr().iterrows():
        for c in row.index:
            if i < c:
                print(i, c, row[c], file=f)
    f.close()

def export_pixels(px, name):
    with open(name, "wb") as f:
        pickle.dump(px, f)

def get_coord_from_name(name):
    return tuple(map(int, re.findall("_c(\d)_(\d).pck", name)[0]))


def iterate_last_axis(arr):
    length = arr.shape[-1]
    for i in range(length):
        yield arr[..., i]

def load_masks(name, root):
    masks_files = glob.glob(str(P(root) / "maskrcnn*.pck"))
    print(f"Fetched {len(masks_files)} mask files")
    masks = []
    for name in masks_files:
        coord = get_coord_from_name(name)
        with open(name, "rb") as f:
            obj = pickle.load(f)
            for mask in iterate_last_axis(obj):
                masks.append((mask, coord))

    return masks


def to_ori(coord):
    return (coord[0] * STEP - MARGIN, coord[1] * STEP - MARGIN)


def get_mask_size(mask):
    return mask.sum()


def get_center_of_mass(mask, coord):
    ori = to_ori(coord)
    com = center_of_mass(mask)
    return (com[0] + ori[0], com[1] + ori[1])


def mask_as_all(mask, coord, shape):
    ori = to_ori(coord)
    arr = np.zeros(shape)
    x_a, x_b = max(0, ori[0]), min(ori[0] + mask.shape[0], shape[0])
    y_a, y_b = max(0, ori[1]), min(ori[1] + mask.shape[1], shape[1])
    mx_a, mx_b = x_a - ori[0], -ori[0] + x_b
    my_a, my_b = y_a - ori[1], -ori[1] + y_b
    arr[x_a:x_b, y_a:y_b] = mask[mx_a:mx_b, my_a:my_b]
    return arr


def mask_is_close(size1, size2, com1, com2):
    size_ok = abs(min(size1, size2) / max(size1, size2)) < 0.4
    com_ok = np.linalg.norm(np.array(com1) - np.array(com2)) < 20
    return size_ok and com_ok


def filter_masks(masks):
    registered = []
    filtered_masks = []
    doublet = 0
    for mask, coord in masks:
        size = get_mask_size(mask)
        com = get_center_of_mass(mask, coord)
        if (
            not any(map(lambda x: mask_is_close(x[0], size, x[1], com), registered))
        ) and size > 20:
            filtered_masks.append((mask, coord))
            registered.append((size, com))
        else:
            doublet += 1

    print(f"We have {doublet} doublet and {len(registered)} kept !")

    return filtered_masks


def flat_depth_mask(masks, shape):
    new_mask = np.zeros(shape[:2], dtype=np.uint64)
    for i, (mask, coord) in enumerate(masks):
        new_mask += (mask_as_all(mask, coord, shape) * i).astype(np.uint64)
    return new_mask


def mask_to_one(masks, shape):
    new_mask = np.zeros(shape[:2], dtype=np.uint64)
    for i, (mask, coord) in enumerate(masks):
        mask = mask_as_all(mask, coord, shape)
        while True:
            idx = random.choice(range(1, 7))
            cover = np.unique(new_mask[mask.nonzero()])
            if not idx in cover:
                new_mask[mask.nonzero()] = (mask[mask.nonzero()] * idx).astype(np.uint64)
                break
            else:
                print("T", end="")
    return new_mask


def show_segmentation(img, flatmask, figpath):
    img.seek_image(t=0, z=0, c=3)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(flatmask == 0)
    plt.subplot(1, 2, 2)
    plt.imshow(img.img)
    plt.savefig(str(figpath))
    plt.close()

def show_segmentation_color(img, onemask, figpath):
    img.seek_image(t=0, z=0, c=3)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(onemask, cmap=cm.Set3)
    plt.subplot(1, 2, 2)
    plt.imshow(img.img)
    plt.savefig(str(figpath))
    plt.close()


def extract_pixels(masks, img):
    nb_channels = img.get_c()
    nb_cells = len(masks)
    ls_pixels = [[None] * nb_channels for i in range(nb_cells)]
    for c in range(nb_channels):
        img.seek_image(t=0, z=0, c=c)
        arr = np.asarray(img.img)
        for i, (mask, coord) in enumerate(masks):
            ls_pixels[i][c] = arr[mask_as_all(mask, coord,
                                              arr.shape).nonzero()]

    return ls_pixels




def robust_mean(x):
    size = x.size
    return np.mean(np.sort(x)[int(0.1 * size) : int(0.9 * size)])


def get_df_scores(pxs):
    dict_df = dict()
    for c in range(4):
        ls = [robust_mean(pxs[d][c]) for d in range(len(pxs))]
        dict_df["color_" + str(c)] = ls
    return pd.DataFrame(dict_df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Image file to be analyzed")

    args = parser.parse_args()
    args.file = P(args.file)
    args.dir = args.file.parent / args.file.stem
    conf_path = args.dir / "maskrcnn_v2.conf"
    conf = ConfigParser()
    conf.read(conf_path)
    for k, v in conf["DEFAULT"].items():
        setattr(args, k, v)

    STEP = int(args.step)
    MARGIN = int(args.margin)
    main(args)
