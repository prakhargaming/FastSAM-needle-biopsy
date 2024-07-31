from time import sleep
from zaber_motion import Library
from zaber_motion.ascii import Connection
from FastSAM_img_segmentation import img_segment  # Adjusted to use absolute import


def move_stages(coordinates, path, connection):
    Library.enable_device_db_store()
    # Ensure there are at least two devices connected
    device_list = connection.detect_devices()
    if len(device_list) < 2:
        print("Not enough devices found.")
        return

    # Assuming the devices are the first two detected
    x_stage = device_list[0].get_axis(1)
    y_stage = device_list[1].get_axis(1)

    xlen = x_stage.get_number_of_index_positions()
    ylen = y_stage.get_number_of_index_positions()

    xinc = xlen//21
    yinc = ylen//21

    # Home both stages before starting the movements
    x_stage.home()
    y_stage.home()

    print("Stages homed.")

    # Debugging: Print the coordinates and path
    print("Coordinates:", coordinates)
    print("Path:", path)

    # Move through the path
    for idx in path:
        # Debugging: Print the current index and type
        print("Current index (path):", idx, "Type:", type(idx))
        x, y = coordinates[idx]
        x = int(x * xinc)
        y = int(y * yinc)
        print(f"Moving to (x: {x}, y: {y})")

        # Move the x stage to the x coordinate
        x_stage.move_index(x)

        # Move the y stage to the y coordinate
        y_stage.move_index(y)

        # Optionally, you can add a small delay to ensure the stages have time to move
        sleep(0.5)

    print("Path traversal complete.")

def main():
    # Connect to the Zaber devices
    with Connection.open_iot('918be735-c159-4687-9fb9-5c060d8712ad', 'RNx9oqwAF776DPU8w2PFRCP5UqYUDq3C') as connection:
        device_list = connection.detect_devices()
        print("Found {} devices".format(len(device_list)))

        pathAndCoords = img_segment()

        coords = pathAndCoords[0]
        path = pathAndCoords[1]

        move_stages(coords, path, connection)

if __name__ == "__main__":
    main()
