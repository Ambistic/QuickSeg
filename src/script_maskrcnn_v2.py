"""
This script is intended to be used as mask maker
using maskrcnn. We use a separate script because mask rcnn
has different dependencies from the rest.
"""

import numpy as np
from tqdm import tqdm
import sys
import argparse
from  configparser import ConfigParser

sys.path.append("../libs")
from tiff import Tiff
from pathlib import Path as P
from roimaker import segment_image, quick_segment
from maskrcnn import get_model
import skimage
import pickle


def prepro(image):
    image = np.asarray(image)
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    return image

def normalize(image):
    thr = np.max(image)
    image = np.asarray(image) * 255. / thr
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def export(obj, output, z, coord):
    dest = P(output) / f"maskrcnn_{z}_c{coord}.pck"
    with open(dest, "wb") as f:
        pickle.dump(obj, f)

def extract_patch(image, i, j, step, size, margin):
    shape_x, shape_y = image.shape[:2]
    i_step, j_step = i * step, j * step

    ref_x_start, ref_y_start = -margin + i_step, -margin + j_step
    ref_x_end, ref_y_end = -margin + i_step + size, -margin + j_step + size

    im_x_start, im_y_start = max(0, ref_x_start), max(0, ref_y_start)
    im_x_end, im_y_end = min(shape_x, ref_x_end), min(shape_y, ref_y_end)

    pa_x_start, pa_y_start = im_x_start - ref_x_start, im_y_start - ref_y_start
    pa_x_end, pa_y_end = size + im_x_end - ref_x_end, size + im_y_end - ref_y_end

    patch = np.zeros((size, size, 3))
    patch[pa_x_start:pa_x_end, pa_y_start:pa_y_end] = \
        image[im_x_start:im_x_end, im_y_start:im_y_end]

    return patch

def test_central_frame(mask, margin, thr=0.2):
    return (mask[margin:-margin, margin:-margin].sum() / mask.sum()) > thr

def extract_rois(result, margin):
    print("SHAPE", result["masks"].shape)
    rois = list(np.rollaxis(result["masks"], 2))
    print("LEN", len(rois))
    filtered_rois = list(filter(lambda x: test_central_frame(x, margin),
                                rois))

    if len(filtered_rois) == 0:
        return np.array([])

    filtered_rois = np.stack(filtered_rois, axis=-1)
    print("FINALLY", filtered_rois.shape)
    return filtered_rois

def cut_and_detect(args, image):
    shape_x, shape_y = image.shape[:2]

    # open model
    model = get_model()

    for i, c_i in enumerate(range(0, shape_x, args.step)):
        for j, c_j in enumerate(range(0, shape_y, args.step)):
            coord = f"{i}_{j}"
            patch = extract_patch(image, i, j, args.step,
                                  args.size, args.margin)
            res = model.detect([patch], verbose=0)[0]
            rois = extract_rois(res, args.margin)
            if not len(rois):
                print(coord, "is empty")
            yield rois, coord


def export_config(args):
    conf = ConfigParser()
    conf["DEFAULT"] = dict(
        image_file=str(args.file),
        time=args.time,
        channel=args.channel,
        z=args.depth,
        output=str(args.output),
        name=args.name,
        step=args.step,
        margin=args.margin,
        size=args.size,
    )
    with open(args.output / 'maskrcnn_v2.conf', 'w') as configfile:
        conf.write(configfile)


def main(args):
    # open image
    img = Tiff(args.file)
    img.get_imagej_metadata()

    # process
    img.seek_image(args.time, args.depth, args.channel)
    image = prepro(img.img)

    print(image.shape)
    for rois, coord in cut_and_detect(args, image):
        export(rois, args.output, args.name, coord)

    export_config(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-z", "--depth", default=0, type=int)
    parser.add_argument("-c", "--channel", default=0, type=int)
    parser.add_argument("-t", "--time", default=0, type=int)
    parser.add_argument("-s", "--size", default=224, type=int)
    parser.add_argument("-m", "--margin", default=20, type=int)

    args = parser.parse_args()

    if args.output is None:
        args.output = P(args.file).parent / P(args.file).stem

    args.output.mkdir(parents=True, exist_ok=True)
    args.name = "t%d_z%d_c%d" % (args.time, args.depth, args.channel)
    args.step = args.size - 2 * args.margin

    main(args)
