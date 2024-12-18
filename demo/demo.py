# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

from datasets.prepare_nassar2020 import prepare_nassar2020

prepare_nassar2020()

# constants
WINDOW_NAME = "mask2former demo"

CONFIG = {
    "length": 512,
    "overlap_length": 256,
    "half_overlap": 128,
}

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
    
def tile_images(image):
    tiles = []
    
    for y in range(0, image.shape[0] - CONFIG["overlap_length"] - 1, CONFIG["overlap_length"]):
        for x in range(0, image.shape[1] - CONFIG["overlap_length"] - 1, CONFIG["overlap_length"]):
            adjustedX = min(image.shape[1] - CONFIG["length"], x)
            adjustedY = min(image.shape[0] - CONFIG["length"], y)
            tiles.append(image[adjustedY:adjustedY + CONFIG["length"], adjustedX:adjustedX + CONFIG["length"]])
            
    return tiles
    
def reconstruct(tiles, original_width, original_height):
    x = 0
    y = 0
    output = np.zeros((original_height, original_width, 3))

    for tile in tiles:
        if x + CONFIG["overlap_length"] >= original_width:
            x = 0
            y += CONFIG["overlap_length"]
        if y + CONFIG["overlap_length"] >= original_height:
            print(f"{y} too large for original height {original_height}")
        
        startX = CONFIG["half_overlap"]
        if x > original_width - CONFIG["length"]:
            startX = CONFIG["length"] - (original_width - x - CONFIG["half_overlap"])

        startY = CONFIG["half_overlap"]
        if y > original_height - CONFIG["length"]:
            startY = CONFIG["length"] - (original_height - y - CONFIG["half_overlap"])
        
        outputStartX = x + CONFIG["half_overlap"]
        if x == 0:
            outputStartX = 0
            startX = 0
            
        outputStartY = y + CONFIG["half_overlap"]
        if y == 0:
            outputStartY = 0
            startY = 0

        output[outputStartY:min(y + CONFIG["length"], original_height), outputStartX:min(x + CONFIG["length"], original_width)] = tile[startY:, startX:]

        x += CONFIG["overlap_length"]
        
    return output


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            
            original_height, original_width, _ = img.shape
            images = tile_images(img)
            mask_tiles = []
            
            start_time = time.time()
            
            for tile in images:
                mask_tiles.append(demo.run_on_image(tile)[1])
                
            reconstructed_mask = reconstruct(mask_tiles, original_width, original_height)

            if args.output and os.path.isdir(args.output):
                out_filename = os.path.join(args.output, os.path.basename(path))
                cv2.imwrite(out_filename, reconstructed_mask)
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
