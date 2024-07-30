import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tsp_solver.greedy import solve_tsp

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

def process_image(img, output_path=None):

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
    # _, _, bw_mask = cv2.split(bw_mask)
    
    if output_path:
        cv2.imwrite(output_path, bw_mask)

    return bw_mask

def resize_image(image, dims):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    resized_image = cv2.resize(image, (dims[0], dims[1]), interpolation=cv2.INTER_AREA)
    
    if len(resized_image.shape) == 3:
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = resized_image
    
    normalized_array = grayscale_image.astype(np.float32) / 255.0
    return normalized_array

def generate_bitmask(bw):
    return (bw > 0).astype(int)

def find_white_pixel_coordinates(bitmask):
    return np.argwhere(bitmask == 1)

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def generate_distance_matrix(coordinates):
    size = len(coordinates)
    distance_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(i + 1, size):
            dist = manhattan_distance(coordinates[i], coordinates[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    
    return distance_matrix

def visualize_path(bitmask, coordinates, path, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(bitmask, cmap='gray')
    
    for i in range(len(path) - 1):
        start = coordinates[path[i]]
        end = coordinates[path[i + 1]]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'r-')
    
    plt.scatter(coordinates[:, 1], coordinates[:, 0], c='blue')
    plt.scatter(coordinates[path[0]][1], coordinates[path[0]][0], c='green', s=100, label='Start', edgecolor='black')
    plt.scatter(coordinates[path[-1]][1], coordinates[path[-1]][0], c='red', s=100, label='End', edgecolor='black')
    
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def visualize_path2(bitmask, coordinates, path):
    """
    Visualizes the path on the bitmask and returns the visualization as a NumPy array.

    Parameters:
        bitmask (np.ndarray): The bitmask image.
        coordinates (np.ndarray): The coordinates of the points.
        path (list): The path of the coordinates.

    Returns:
        np.ndarray: The visualization as an image array.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(bitmask, cmap='gray')
    
    for i in range(len(path) - 1):
        start = coordinates[path[i]]
        end = coordinates[path[i + 1]]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'r-')
    
    plt.scatter(coordinates[:, 1], coordinates[:, 0], c='blue')
    plt.scatter(coordinates[path[0]][1], coordinates[path[0]][0], c='green', s=100, label='Start', edgecolor='black')
    plt.scatter(coordinates[path[-1]][1], coordinates[path[-1]][0], c='red', s=100, label='End', edgecolor='black')
    
    plt.legend()

    # Convert the plot to a NumPy array
    plt.draw()
    image_array = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    return image_array


def post_process_path(coordinates, path):
    new_path = [path[0]]
    new_coordinates = coordinates.copy()
    for i in range(1, len(path)):
        current = coordinates[path[i-1]]
        next = coordinates[path[i]]
        
        if current[0] != next[0]:
            new_path.append(len(new_coordinates))
            new_coordinates = np.vstack((new_coordinates, [next[0], current[1]]))
        
        new_path.append(path[i])
    
    return new_coordinates, new_path

def travelling_salesman(image, output_path, resize_dims=(21, 21), visualize=False):
    bw = resize_image(image, resize_dims)
    bitmask = generate_bitmask(bw)
    
    # Find coordinates and solve TSP
    coordinates = find_white_pixel_coordinates(bitmask)
    distance_matrix = generate_distance_matrix(coordinates)
    path = solve_tsp(distance_matrix)
    
    # Post-process the path
    coordinates, path = post_process_path(coordinates, path)
    
    vizualization = visualize_path(bitmask, coordinates, path, output_path)
    
    return coordinates, path, vizualization