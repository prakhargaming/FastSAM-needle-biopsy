import os
import ast
import torch
import numpy as np
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt
from utils.tools import convert_box_xywh_to_xyxy
from os import listdir
from os.path import isfile, join, isdir
import argparse
import sys
import warnings
import json
import cv2

def convert_to_list(obj):
    """Recursively convert numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_list(item) for item in obj)
    else:
        return obj
    
def crop_image(image_path, top=100, bottom=100, left=100, right=100):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    cropped_img = img[top:height-bottom, left:width-right]
    return cropped_img

def custom_formatwarning(msg, *args, **kwargs):
    return f"WARNING: {str(msg)}\n"

warnings.formatwarning = custom_formatwarning

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    print(f"WARNING: {message}", file=sys.stderr)

warnings.showwarning = custom_showwarning

# Redirect warnings to stderr
warnings.showwarning = lambda *args, **kwargs: print(warnings.formatwarning(*args, **kwargs), file=sys.stderr)

# Use this for actual errors
def print_error(msg):
    print(f"ERROR: {msg}", file=sys.stderr)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_files(parent_dir):
    parent_dir = os.path.normpath(parent_dir)  
    return [os.path.join(parent_dir, f) for f in listdir(parent_dir) if isfile(join(parent_dir, f))]

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default="./weights/FastSAM-x.pt")
        parser.add_argument("--img_path", type=str, default="./tissue/")
        parser.add_argument("--imgsz", type=int, default=1024)
        parser.add_argument("--iou", type=float, default=0.7)
        parser.add_argument("--conf", type=float, default=0.75)
        parser.add_argument("--output", type=str, default="./output/")
        parser.add_argument("--point_prompt", type=str, default="[[0,0]]")
        parser.add_argument("--point_label", type=str, default="[0]")
        parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]")
        parser.add_argument("--better_quality", type=bool, default=False)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--retina", type=bool, default=True)
        parser.add_argument("--withContours", type=bool, default=False)
        parser.add_argument("--microDims", type=str, default="21,21")
        parser.add_argument("--plot", type=bool, default=True)
        return parser.parse_args()

def img_segment(
    model_path="./weights/FastSAM-x.pt",
    img_path="./tissue/",
    imgsz=1024,
    iou=0.7,
    conf=0.75,
    output="./output/",
    point_prompt="[[0,0]]",
    point_label="[0]",
    box_prompt="[[0,0,0,0]]",
    better_quality=False,
    device=None,
    retina=True,
    withContours=False,
    microDims="35,27",
    plot=True
):
    """
    Segments images and charts the shortest path using the FastSAM model and saves the segmented images to the output directory.

    Parameters:
        model_path (str): Path to the model weights file.
        img_path (str): Path to the image file or directory containing images.
        imgsz (int): Image size for processing.
        iou (float): IOU threshold for filtering the annotations.
        conf (float): Object confidence threshold.
        output (str): Directory to save the segmented images.
        point_prompt (str): Points prompt in the format "[[x1,y1],[x2,y2]]".
        point_label (str): Point labels in the format "[1,0]" (0: background, 1: foreground).
        box_prompt (str): Box prompt in the format "[[x,y,w,h],[x2,y2,w2,h2]]".
        better_quality (bool): Flag to use better quality using morphologyEx.
        device (str): Device to run the model on ("cuda", "mps", "cpu").
        retina (bool): Flag to draw high-resolution segmentation masks.
        withContours (bool): Flag to draw the edges of the masks.
        microDims (str): Dimensions of image resize for shortest path in the format "width,height".
        plot (bool): Flag to save and return plot of the results.
        
    Returns:
        coords (list[tuple(int, int)]): list of tuple coordinates to visit
        path (list[int]): list of integers representing what order to visit the coords in 
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    if isdir(img_path):
        files_list = get_files(img_path)
    else:
        files_list = [img_path]

    model = FastSAM(model_path)
    point_prompt = ast.literal_eval(point_prompt)
    box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(box_prompt))
    point_label = ast.literal_eval(point_label)
    microDims = ast.literal_eval("(" + microDims + ")")

    for file_path in files_list:
        try:
            input_img_cv = cv2.imread(file_path)

            crop_pixels = 100
        
            height, width = input_img_cv.shape[:2]
            input_img_cv = input_img_cv[crop_pixels:height-crop_pixels, crop_pixels:width-crop_pixels]

            # Convert OpenCV image to PIL Image
            input_img = Image.fromarray(cv2.cvtColor(input_img_cv, cv2.COLOR_BGR2RGB))

            everything_results = model(
                input_img,
                device=device,
                retina_masks=retina,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
            )
            bboxes = None
            points = None
            prompt_process = FastSAMPrompt(input_img, everything_results, device=device)
            if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
                ann = prompt_process.box_prompt(bboxes=box_prompt)
                bboxes = box_prompt
            elif point_prompt[0] != [0, 0]:
                ann = prompt_process.point_prompt(
                    points=point_prompt, pointlabel=point_label
                )
                points = point_prompt
                point_label = point_label
            else:
                ann = prompt_process.everything_prompt()

            output_path = os.path.join(output, os.path.basename(file_path))

            path, coordinates = prompt_process.plot(
                annotations=ann,
                output_path=output_path,
                bboxes=bboxes,
                points=points,
                point_label=point_label,
                withContours=withContours,
                better_quality=better_quality,
                microDims=microDims,
                plot=plot
            )
        except Exception as e:
            eprint("Error processing this image. Please use the manual scanning feature.")
            eprint(e)
            return

    path = convert_to_list(path)
    coordinates = convert_to_list(coordinates)

    ordered_coords = [coordinates[i] for i in path]

    print(ordered_coords)

    return json.dumps({"path": path, "coordinates": ordered_coords})

if __name__ == "__main__":
    args = parse_args()
    img_segment(
        model_path=args.model_path,
        img_path=args.img_path,
        imgsz=args.imgsz,
        iou=args.iou,
        conf=args.conf,
        output=args.output,
        point_prompt=args.point_prompt,
        point_label=args.point_label,
        box_prompt=args.box_prompt,
        better_quality=args.better_quality,
        device=args.device,
        retina=args.retina,
        withContours=args.withContours,
        microDims=args.microDims,
        plot=args.plot
    )
