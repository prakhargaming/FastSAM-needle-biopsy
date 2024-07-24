import cv2
import numpy as np

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

    cv2.imwrite(output_path, bw_mask)