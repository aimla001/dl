"""
Aim: To perform basic image processing operations using Python and OpenCV and analyze the effects of histogram equalization,
edge detection, thresholding, image augmentation, and morphological operations.

Algorithm:

1. Import the required libraries and load the input image.
2. Apply histogram equalization to enhance the image contrast.
3. Compute and display the histograms of the original and equalized images.
4. Apply Gaussian Blur and perform Canny edge detection.
5. Perform image augmentation using rotation, zoom and horizontal flipping.
6. Convert the image into a binary image using thresholding.
7. Apply erosion and dilation using a structuring kernel.
8. Display the original and all processed images for comparison.

Result:

Thus, basic image processing operations were successfully implemented using Python and OpenCV, producing the expected enhanced
and transformed images.
"""

import cv2, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = r"Image_1.jfif"
img = cv2.imread(path, 0)

def show(imgs, titles, rows, cols, size=(10, 8)):
    plt.figure(figsize=size)
    for i, (im, t) in enumerate(zip(imgs, titles), 1):
        plt.subplot(rows, cols, i); plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")
    plt.tight_layout(); plt.show()

equ = cv2.equalizeHist(img)
show([img, equ], ["Original", "Equalized"], 1, 2)
for i, (im, t) in enumerate([(img, "Original"), (equ, "Equalized")], 1):
    plt.subplot(1, 2, i); plt.hist(im.ravel(), 256, [0, 256]); plt.title(t + " Histogram")
plt.show()

edges = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 100, 200)
show([img, edges], ["Original", "Edges"], 1, 2)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
k = np.ones((5, 5), np.uint8)
show([img, binary, cv2.erode(binary, k), cv2.dilate(binary, k)],
     ["Original", "Binary Threshold", "Erosion", "Dilation"], 2, 2)

rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)[None]
gen = ImageDataGenerator(rotation_range=40, zoom_range=0.2, horizontal_flip=True).flow(rgb, batch_size=1)
show([rgb[0]] + [next(gen)[0].astype("uint8") for _ in range(8)],
     ["Original"] + [f"Augmented {i}" for i in range(1, 9)], 3, 3, (10, 10))