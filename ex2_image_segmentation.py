import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from ex1_image_denoising import space_filter

# Prewitt
# Detect the vertical edge
prewitt_fil_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

# Detect horizontal edges
prewitt_fil_y = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]])

# Sobel
# Detect the vertical edge
sobel_fil_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

# Detect horizontal edges
sobel_fil_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

# Laplacian
# 4-nei
laplacian_fil_4 = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])

# 8-nei
laplacian_fil_8 = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])

# fill each class of the image with different colors
def image_color_fill(label_img):
    assert np.max(label_img) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize the gray image
    visual_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    for i in range(visual_img.shape[0]):  # i for h
        for j in range(visual_img.shape[1]):
            color = label2color_dict[label_img[i, j]]
            visual_img[i, j, 0] = color[0]
            visual_img[i, j, 1] = color[1]
            visual_img[i, j, 2] = color[2]

    return visual_img


img = cv2.imread("test3.jpg")   # read the image
# change the image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img)
img_type = img.shape
print(img_type)
# cv2.imshow("img_gray",img)
# cv2.waitKey()
plt.imshow(img, cmap="gray")  # show the image
plt.show()
res2 = space_filter(img, fil_type="GRAY", fil=laplacian_fil_8, mode="SAME")
plt.imshow(res2, cmap="gray")
plt.imsave("res2.jpg", res2)
print(res2.shape)
plt.show()
