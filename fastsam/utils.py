import numpy as np
import torch
import cv2
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    '''Adjust bounding boxes to stick to image border if they are within a certain threshold.
    Args:
    boxes: (n, 4)
    image_shape: (height, width)
    threshold: pixel threshold
    Returns:
    adjusted_boxes: adjusted bounding boxes
    '''

    # Image dimensions
    h, w = image_shape

    # Adjust boxes
    boxes[:, 0] = torch.where(boxes[:, 0] < threshold, torch.tensor(
        0, dtype=torch.float, device=boxes.device), boxes[:, 0])  # x1
    boxes[:, 1] = torch.where(boxes[:, 1] < threshold, torch.tensor(
        0, dtype=torch.float, device=boxes.device), boxes[:, 1])  # y1
    boxes[:, 2] = torch.where(boxes[:, 2] > w - threshold, torch.tensor(
        w, dtype=torch.float, device=boxes.device), boxes[:, 2])  # x2
    boxes[:, 3] = torch.where(boxes[:, 3] > h - threshold, torch.tensor(
        h, dtype=torch.float, device=boxes.device), boxes[:, 3])  # y2

    return boxes

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def bbox_iou(box1, boxes, iou_thres=0.9, image_shape=(640, 640), raw_output=False):
    '''Compute the Intersection-Over-Union of a bounding box with respect to an array of other bounding boxes.
    Args:
    box1: (4, )
    boxes: (n, 4)
    Returns:
    high_iou_indices: Indices of boxes with IoU > thres
    '''
    boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    # obtain coordinates for intersections
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])

    # compute the area of intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # compute the area of both individual boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # compute the area of union
    union = box1_area + box2_area - intersection

    # compute the IoU
    iou = intersection / union  # Should be shape (n, )
    if raw_output:
        if iou.numel() == 0:
            return 0
        return iou

    # get indices of boxes with IoU > thres
    high_iou_indices = torch.nonzero(iou > iou_thres).flatten()

    return high_iou_indices

def image_to_np_ndarray(image):
    if type(image) is str:
        return np.array(Image.open(image))
    elif issubclass(type(image), Image.Image):
        return np.array(image)
    elif type(image) is np.ndarray:
        return image
    return None

def process_image(output_path, img):

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = (36, 25, 25)
    upper_green = (70, 255, 255)

    # Create a mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Create the black and white mask
    bw_mask = np.zeros_like(img[:,:,0])  # Create a single channel image
    bw_mask[mask > 0] = 255  # Set green pixels to white (255)
    
    bw_mask = cv2.GaussianBlur(bw_mask,(31,31),0)
    bw_mask = cv2.threshold(bw_mask, 100, 255, cv2.THRESH_BINARY)[1]
    bw_mask = cv2.cvtColor(bw_mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # cv2.imwrite(output_path, bw_mask)
    
    return bw_mask

def resize_image(image, dims):
    print(dims)
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    
    resized_image = cv2.resize(image, (dims[0], dims[1]), interpolation=cv2.INTER_AREA)
    
    # Convert the resized image to grayscale if it's not already
    if len(resized_image.shape) == 3:
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = resized_image
    
    # Normalize the array to have values between 0 and 1
    normalized_array = grayscale_image.astype(np.float32) / 255.0
    
    return normalized_array

def find_shortest_path(mask, dims=(50,50)):
    bitmap = resize_image(mask, dims)
    
    # Find coordinates of white pixels
    white_pixels = np.argwhere(bitmap > 0.5)  # Threshold at 0.5 for binary classification
    
    if len(white_pixels) <= 1:
        return white_pixels
    
    # Create a grid to keep track of visited cells
    grid = np.zeros(dims, dtype=bool)
    for pixel in white_pixels:
        grid[pixel[0], pixel[1]] = True
    
    # Use a modified nearest neighbor algorithm to find an approximate shortest path
    path = [white_pixels[0]]  # Start with the first white pixel
    current = path[-1]
    
    while len(path) < len(white_pixels):
        # Get valid neighboring cells
        neighbors = [
            (current[0]+1, current[1]),
            (current[0]-1, current[1]),
            (current[0], current[1]+1),
            (current[0], current[1]-1)
        ]
        
        # Filter out neighbors that are out of bounds or not white pixels
        valid_neighbors = [
            n for n in neighbors 
            if 0 <= n[0] < dims[0] and 0 <= n[1] < dims[1] and grid[n[0], n[1]]
        ]
        
        if not valid_neighbors:
            # If no valid neighbors, find the nearest unvisited white pixel
            unvisited = set(map(tuple, white_pixels)) - set(map(tuple, path))
            if not unvisited:
                break
            distances = cdist([current], list(unvisited))
            nearest = np.argmin(distances[0])
            next_pixel = list(unvisited)[nearest]
        else:
            # Choose the nearest valid neighbor
            distances = cdist([current], valid_neighbors)
            nearest = np.argmin(distances[0])
            next_pixel = valid_neighbors[nearest]
        
        path.append(next_pixel)
        current = next_pixel
        grid[current[0], current[1]] = False  # Mark as visited
    
    return np.array(path)

def draw_grid(img, microDims, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = microDims
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def visualize_path(output_path, original_image, path, resize_dims):
    # Create a copy of the original image for visualization
    vis_image = original_image.copy()
    
    # If the image is grayscale, convert it to RGB
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    
    # Get the dimensions of the original image and the resized image
    orig_height, orig_width = original_image.shape[:2]
    resized_height, resized_width = resize_dims[0], resize_dims[1] # As per the resize_image function
    
    # Scale factors
    scale_y = orig_height / resized_height
    scale_x = orig_width / resized_width
    
    # Function to scale coordinates
    def scale_coord(coord):
        return (int(coord[1] * scale_x), int(coord[0] * scale_y))
    
    # Draw the path on the image
    for i in range(len(path) - 1):
        start_point = scale_coord(path[i])
        end_point = scale_coord(path[i+1])
        cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)
    
    # Mark the start and end points
    cv2.circle(vis_image, scale_coord(path[0]), 5, (255, 0, 0), -1)  # Start point in blue
    cv2.circle(vis_image, scale_coord(path[-1]), 5, (0, 0, 255), -1) 

    cv2.imwrite(output_path, draw_grid(vis_image, resize_dims))
    