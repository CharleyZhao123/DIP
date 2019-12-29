import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from ex1_image_denoising import space_filter
import filters

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


def image_get_label(img, mode="THRESHOLD_OSTU"):
    img_h = img.shape[0]
    img_w = img.shape[1]
    label_img = np.zeros((img_h, img_w), dtype="uint16")

    # OSTU
    if mode == "THRESHOLD_OSTU":
        # ret, label_img = cv2.threshold(img, 0, 5, cv2.THRESH_OTSU)  # use opencv

        gray_list = np.zeros(256, dtype="uint16")  # statistical gray value
        # statistical gray probability
        gray_probability = np.zeros(256, dtype="float32")
        pixel_sum = img_h*img_w

        for i in range(img_h):
            for j in range(img_w):
                gray_value = img[i][j]
                gray_list[gray_value] += 1

        # print(gray_list)
        gray_probability = gray_list/pixel_sum
        # print(gray_probability)

        gray_mean = 0
        for i in range(256):
            gray_mean += gray_probability[i]*i

        # otsu
        max_variance = 0.0
        max_T = 0.0
        background_p = 0.0
        background_m = 0.0
        background_s = 0.0
        object_p = 0.0
        object_m = 0.0
        object_s = 0.0
        for T in range(256):
            for i in range(256):
                if i <= T:
                    background_p += gray_probability[i]
                    background_m += i*gray_probability[i]
                else:
                    object_p += gray_probability[i]
                    object_m += i*gray_probability[i]

            if background_p == 0 or object_p == 0:
                continue
            background_u = float(background_m)/background_p
            object_u = float(object_m)/object_p

            for i in range(256):
                if i <= T:
                    background_s += (i-background_u)*(i-background_u) * \
                        gray_probability[i]/background_p
                else:
                    object_s += (i-object_u)*(i-object_u) * \
                        gray_probability[i]/object_u
            W = background_p*background_s*background_s+object_p*object_s*object_s
            B = background_p*(background_u-gray_mean)*(
                background_u-gray_mean)+object_p*(object_u-gray_mean)*(object_u-gray_mean)
            T_variance = B*B/(B*B+W*W)
            if T_variance >= max_variance:
                max_variance = T_variance
                max_T = T

        for i in range(img_h):
            for j in range(img_w):
                gray_value = img[i][j]
                if gray_value <= max_T:
                    label_img[i][j] = 5
                else:
                    label_img[i][j] = 1
        print(label_img)

    elif mode == "EDGE_BASED":
        # Gaussian smoothing
        smooth_img = space_filter(img, fil_type="GRAY",
                                fil=filters.gaussian_fil_5x5, mode="SAME")
        plt.imsave("smooth_img.jpg", smooth_img)
        # get the edge
        edge_img = space_filter(smooth_img, fil_type="GRAY",
                                fil=filters.laplacian_fil_8, mode="SAME")
        plt.imsave("edge_img.jpg", edge_img)
        for i in range(img_h):
            for j in range(img_w):
                gray_value = edge_img[i][j]
                if gray_value >= 30:
                    label_img[i][j] = 5
                else:
                    label_img[i][j] = 1

    return label_img


# fill each class of the image with different colors
def image_color_fill(label_img):
    assert np.max(
        label_img) <= 7, "only 7 classes are supported, add new color in label2color_dict"
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
    visual_img = np.zeros(
        (label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    for i in range(visual_img.shape[0]):  # i for h
        for j in range(visual_img.shape[1]):
            color = label2color_dict[label_img[i, j]]
            visual_img[i, j, 0] = color[0]
            visual_img[i, j, 1] = color[1]
            visual_img[i, j, 2] = color[2]

    return visual_img


img = cv2.imread("test4.jpg")   # read the image
# change the image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img)
img_type = img.shape
print(img_type)
# cv2.imshow("img_gray",img)
# cv2.waitKey()
plt.imshow(img, cmap="gray")  # show the image-=
plt.show()
# plt.imshow(res2, cmap="gray")
label_image = image_get_label(img, mode="EDGE_BASED")
res2 = image_color_fill(label_image)
plt.imsave("res2.jpg", res2)
print(res2.shape)
plt.show()
