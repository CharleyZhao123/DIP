import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from ex1_image_denoising import space_filter
import filters


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

    elif mode == "HIS_MODE":
        gray_list_0 = np.zeros(256, dtype="uint16")  # statistical gray value
        gray_list_1 = np.zeros(256, dtype="uint16")
        gray_list_2 = np.zeros(256, dtype="uint16")

        # statistical gray probability
        for i in range(img_h):
            for j in range(img_w):
                gray_value = img[i][j]
                gray_list_0[gray_value] += 1

        # mean smooth
        for i in range(5, 250):
            gray_list_1[i] = np.mean(gray_list_0[i-5:i+5])

        # Gaussianfilter smooth [1, 2, 1]
        for i in range(256):
            if i == 0:
                gray_list_2[i] = (2*gray_list_1[i]+gray_list_1[i+1])/5
            elif i == 255:
                gray_list_2[i] = (2*gray_list_1[i]+gray_list_1[i-1])/5
            else:
                gray_list_2[i] = (gray_list_1[i-1]+2 *
                                  gray_list_1[i]+gray_list_1[i+1])/5

        his_max = []
        # his_max_ = []
        T = []
        begin = 0
        end = len(gray_list_2)
        for i in range(250):
            if gray_list_2[i-1] < gray_list_2[i] < gray_list_2[i+1] and gray_list_2[i-2] < gray_list_2[i] < gray_list_2[i+2]:
                his_max.append(i)

        for i in range(len(his_max)-1):
            mean = 0.0
            if abs(his_max[i]-his_max[i+1]) > 10 or i == len(his_max)-2:
                end = i
                for j in range(end-begin+1):
                    x = begin+j
                    mean += his_max[x]
                mean = mean/(end-begin+1)
                T.append(mean)
                begin = end+1
                # his_max_.append(his_max[i])
                # his_max_.append(his_max[i+1])

        # for i in range(len(his_max_)-1):
        #     T.append((float(his_max_[i])+his_max_[i])/2)
        # print(T)

        for i in range(img_h):
            for j in range(img_w):
                gray_value = img[i][j]
                for t in range(len(T)):
                    if gray_value <= T[t]:
                        label_img[i][j] = t+1
                        break

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
                if gray_value >= 10:
                    label_img[i][j] = 0
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


img = cv2.imread("org1.jpg")   # read the image
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
label_image = image_get_label(img, mode="HIS_MODE")
res2 = image_color_fill(label_image)
plt.imsave("res2.jpg", res2)
print(res2.shape)
plt.show()
