export default function handler(req, res) {
  res.send(`

import cv2
import os
from google.colab.patches import cv2_imshow
import numpy as np

def resize_mat(layer) :
  new_rows = layer.shape[0] // 2
  new_cols = layer.shape[1] // 2

  convertedMat = np.zeros((new_rows, new_cols), dtype=np.uint8)

  for i in range(new_rows):
    for j in range(new_cols):
      row_start = i * 2
      row_end = row_start + 2
      col_start = j * 2
      col_end = col_start + 2

      block = layer[row_start:row_end, col_start:col_end]
      convertedMat[i][j] = np.average(block)

  return convertedMat


image_path = '/content/lena.jpeg'

if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    if original_image is not None:
      row=original_image.shape[0]
      col=original_image.shape[1]
      depth=original_image.shape[2]
      print("Original Image Dimension : ",row,col,depth)
      red=original_image[:,:,2]
      green=original_image[:,:,1]
      blue=original_image[:,:,0]

      resized_red = resize_mat(red)
      resized_green = resize_mat(green)
      resized_blue = resize_mat(blue)

      # resized_image = cv2.merge((resized_blue, resized_green, resized_red))
      resized_image = np.stack((resized_red, resized_green, resized_blue), axis=-1)

      cv2_imshow(resized_image)
      print("Converted Image dimension :",resized_image.shape)


    else:
        print(f"Failed to read image at {image_path}")

else:
    print(f"File not found: {image_path}")

import cv2
import os
from google.colab.patches import cv2_imshow
import numpy as np

def convertTGrayScale(image):
  GI = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
  for i in range(image.shape[0]) :
    for j in range(image.shape[1]) :
      GI[i][j] = np.average(image[i][j][0:3])
  return GI


image_path = '/content/lena.jpeg'
if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    cv2_imshow(original_image)
    print("Shape of the original : ",original_image.shape)

    if original_image is not None:
        GI = convertTGrayScale(original_image)
        print("Converted image ")
        cv2_imshow(GI)
        print("Shape of the converted : ",GI.shape)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"Failed to read image at {image_path}")

else:
    print(f"File not found: {image_path}")

#clockwise
import cv2
import os
from google.colab.patches import cv2_imshow
import numpy as np

def flip90(image):
  depth = image.shape[2]
  I90 = np.zeros((image.shape[0], image.shape[1] , depth), dtype=np.uint8)
  r = image.shape[0]
  c = image.shape[1]
  for i in range(r) :
    for j in range(c) :
      I90[i][j] = image[r-j-1][i]
  return I90


image_path = '/content/lena.jpeg'
if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    cv2_imshow(original_image)
    print("Shape of the original : ",original_image.shape)

    if original_image is not None:
        I90 = flip90(original_image)
        print("Image 90 flipped")
        cv2_imshow(I90)
        print("Shape of the converted : ",I90.shape)


    else:
        print(f"Failed to read image at {image_path}")

else:
    print(f"File not found: {image_path}")

#clockwise
import cv2
import os
from google.colab.patches import cv2_imshow
import math
import numpy as np

def flip60(image):
  depth = image.shape[2]
  I60 = np.zeros((image.shape[0]+1000, image.shape[1]+1000 , depth), dtype=np.uint8)
  r = image.shape[0]
  c = image.shape[1]
  for i in range(r) :
    for j in range(c) :
      x = i*math.cos(60) - j*math.sin(60)
      y = i*math.sin(60) + j*math.cos(60)
      I60[x][y] = image[i][j]
  return I60

image_path = '/content/lena.jpeg'
if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    cv2_imshow(original_image)
    print("Shape of the original : ",original_image.shape)

    if original_image is not None:
        I60 = flip60(original_image)
        print("Image 180 flipped")
        cv2_imshow(I60)
        print("Shape of the converted : ",I60.shape)


    else:
        print(f"Failed to read image at {image_path}")

else:
    print(f"File not found: {image_path}")

import cv2
import os
from google.colab.patches import cv2_imshow
import math
import numpy as np

def fliptheta(image, theta):
    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, -theta, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    return rotated

# Path to the image
image_path = '/content/lena.jpeg'
if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    cv2_imshow(original_image)
    print("Shape of the original : ", original_image.shape)

    if original_image is not None:
        I60 = fliptheta(original_image, 60)
        print("Image rotated 60 degrees clockwise")
        cv2_imshow(I60)
        print("Shape of the rotated : ", I60.shape)

    else:
        print(f"Failed to read image at {image_path}")
else:
    print(f"File not found: {image_path}")

import cv2
import os
from google.colab.patches import cv2_imshow
import math
import numpy as np

def translate(image,tx,ty):
    r = image.shape[0]
    c = image.shape[1]
    rotated = np.zeros((r+tx,c+ty,3), dtype=np.uint8)
    for i in range(r) :
      for j in range(c) :
        rotated[i+tx][j+ty] = image[i][j]

    return rotated

# Path to the image
image_path = '/content/lena.jpeg'
if os.path.exists(image_path):
    original_image = cv2.imread(image_path)
    cv2_imshow(original_image)
    print("Shape of the original : ", original_image.shape)

    if original_image is not None:
        translatedImage  = translate(original_image, 10,20)
        print("Image translated")
        cv2_imshow(translatedImage)
        print("Shape of the translated : ", translatedImage.shape)

    else:
        print(f"Failed to read image at {image_path}")
else:
    print(f"File not found: {image_path}")


# Image Enlarge , inter polation

import cv2
from google.colab.patches import cv2_imshow

image_path = 'bestimage.jpg'

img = cv2.imread(image_path)

if img is not None:
    print("Original size:", img.shape)
    cv2_imshow(img)

    scale = 2
    height, width = img.shape[:2]
    new_size = (int(width * scale), int(height * scale))

    bigger_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    print("New size:", bigger_img.shape)
    cv2_imshow(bigger_img)

else:
    print("Image not found.")

`);
}
