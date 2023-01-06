import os
import cv2
import numpy as np

def save_image(image, save_path, name):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, name), image)