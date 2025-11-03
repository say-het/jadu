export default function handler(req, res) {
  res.send(`
import cv2
import os
from google.colab.patches import cv2_imshow
import numpy as np

def cvtGray(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale

def image_negatives(image):
    negative = 255 - image
    return negative

image_path = '/content/lena.jpeg'

if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    print(original_image.shape)

    if original_image is not None:
        grayscale_image = cvtGray(original_image)
        image_negative = image_negatives(grayscale_image)
        image_negative_original = image_negatives(original_image)
        cv2_imshow(original_image)
        cv2_imshow(image_negative_original)
        cv2_imshow(grayscale_image)
        cv2_imshow(image_negative)

    else:
        print(f"Failed to read image at {image_path}")

else:
    print(f"File not found: {image_path}")

import cv2
import os
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np

def power_law_transform(image, gamma=1.5):
    image_float = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    power_transformed = np.power(image_float, gamma)
    power_transformed = power_transformed * 255  # Scale back to [0,255]
    power_transformed = np.clip(power_transformed, 0, 255)
    power_transformed = power_transformed.astype(np.uint8)
    return power_transformed

image_path = '/content/log_transform.jpg'

if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    print(original_image.shape)

    if original_image is not None:
        gamma = 0.5  # Try gamma = 0.4 too
        power_transform_image = power_law_transform(original_image, gamma=gamma)

        cv2_imshow(original_image)
        cv2_imshow(power_transform_image)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(original_image.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
        plt.title("Original Intensity Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(power_transform_image.ravel(), bins=256, range=(0, 255), color='purple', alpha=0.7)
        plt.title(f"Power-Transformed Histogram (gamma={gamma})")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    else:
        print(f"Failed to read image at {image_path}")
else:
    print(f"File not found: {image_path}")

import cv2
import os
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np

def log_transform(image):
    image_float = image.astype(np.float32)

    c = 255 / np.log(1 + np.max(image_float))
    log_transformed = c * np.log(1 + image_float)

    log_transformed = np.clip(log_transformed, 0, 255)
    log_transformed = log_transformed.astype(np.uint8)

    return log_transformed

image_path = '/content/log_transform.jpg'

if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    print(original_image.shape)

    if original_image is not None:
        log_transformImage = log_transform(original_image)
        cv2_imshow(original_image)
        cv2_imshow(log_transformImage)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(original_image.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
        plt.title("Original Intensity Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(log_transformImage.ravel(), bins=256, range=(0, 255), color='green', alpha=0.7)
        plt.title("Log-Transformed Intensity Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()


    else:
        print(f"Failed to read image at {image_path}")

else:
    print(f"File not found: {image_path}")

import os
from google.colab.patches import cv2_imshow
import numpy as np


image_path = '/content/bestimage.jpg'
def contrastStretch(image, new_min=0, new_max=40):
    min_val = np.min(image)
    max_val = np.max(image)

    stretched = ((image - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return stretched.astype(np.uint8)


if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    print(original_image.shape)

    if original_image is not None:
        contrast_stretched_image = contrastStretch(original_image)
        cv2_imshow(original_image)
        cv2_imshow(contrast_stretched_image)

    else:
        print(f"Failed to read image at {image_path}")

else:
    print(f"File not found: {image_path}")

`);
}
