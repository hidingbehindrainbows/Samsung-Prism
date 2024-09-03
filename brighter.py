import os

import cv2
import numpy as np


def gamma_correction(image, gamma=1.0):
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


folder_path = 'C:/mycode/python/Samsung Prism/All Datasets/Dark And Bright Dataset/dark'
output_folder_path = 'C:/mycode/python/Samsung Prism/All Datasets/Dark And Bright Dataset/bright'

files = os.listdir(folder_path)

for i, filename in enumerate(files, start=353):
    try:
        # Load image
        image = cv2.imread(os.path.join(folder_path, filename))

        if image is None:
            print(f"Unable to read file: {filename}")
            continue

        # Apply gamma correction
        brightened_image = gamma_correction(image, gamma=1.6)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2LAB)

        # Apply CLAHE on the L channel
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        # Merge modified LAB channels
        modified_lab_image = cv2.merge((l_channel, a_channel, b_channel))

        # Convert LAB back to BGR
        result_image = cv2.cvtColor(modified_lab_image, cv2.COLOR_LAB2BGR)

        # Blend images
        alpha = 0.9
        result_blend = cv2.addWeighted(
            brightened_image, alpha, result_image, 1 - alpha, alpha)

        # Save result
        cv2.imwrite(os.path.join(output_folder_path, filename), result_blend)
        print(f"Processed file: {filename}")

    except Exception as e:
        print(f"Error processing file: {filename} - {e}")
