import cv2
import numpy as np

def is_orange(pixel, tolerance=50):
    b, g, r = pixel  # OpenCV uses BGR color order
    # Check if the pixel is more red than green, and more green than blue
    return r > g > b and r > 100 and g > 50 and b < 100

def process_image(output_path, image):

    cv2.imwrite("orange" + output_path, image)

    # Create a mask where orange pixels are 255, others are 0
    orange_mask = np.apply_along_axis(is_orange, 2, image).astype(np.uint8) * 255
    
    # Create the output image (black background)
    output_image = np.zeros_like(image)
    
    # Set orange pixels to white (255, 255, 255)
    output_image[orange_mask == 255] = [255, 255, 255]
    
    # Save the output image
    cv2.imwrite(output_path, output_image)
    
    print(f"Processed image saved to {output_path}")

# Example usage
# input_image_path = 'path/to/your/input/image.jpg'
# output_image_path = 'path/to/your/output/image.jpg'
# process_image(input_image_path, output_image_path)