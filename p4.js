export default function handler(req, res) {
  res.send(`

import numpy as np
import cv2
from matplotlib import pyplot as plt

img_path = '/content/hitsogram_image_e.jpg'
img9 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

pdf = np.zeros(256, dtype=int)
cdf = np.zeros(256, dtype=int)
normalized_cdf = np.zeros(256, dtype=int)
outimg = np.zeros_like(img9, dtype=np.uint8)

for i in range(img9.shape[0]):
    for j in range(img9.shape[1]):
        pdf[img9[i, j]] += 1

cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]
#normalisation of cdf
# x* y =total
# 255* cdf /total
total_pixels = img9.shape[0] * img9.shape[1]
for i in range(256):
    normalized_cdf[i] = (255 * cdf[i]) // total_pixels

for i in range(img9.shape[0]):
    for j in range(img9.shape[1]):
        outimg[i, j] = normalized_cdf[img9[i, j]]


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img9, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(outimg, cmap='gray')
plt.title("Histogram Equalized Image")
plt.axis("off")

plt.tight_layout(pad=4)
plt.show()


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(img9.flatten(), bins=256, range=(0, 256), color='r', density=True)
plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.subplot(1, 2, 2)
plt.hist(outimg.flatten(), bins=256, range=(0, 256), color='g', density=True)
plt.title("Histogram of Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.tight_layout(pad=4)
plt.show()

# from cv2 import cv2_imshow
from google.colab.patches import cv2_imshow
dollor_path = 'dollar.jpeg'
dollor = cv2.imread(dollor_path)

for i in range(7, -1, -1):
    bit_img = dollor // (2**i) * (2**i)
    cv2_imshow(bit_img)
`);
}
