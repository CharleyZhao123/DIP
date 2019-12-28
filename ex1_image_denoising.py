import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from image_evaluation import psnr

# a convolution kernel sample
# mean filter
fil_sample = 1/9*np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

# Gaussian filter
gaussian_fil_3x3 = 1/16*np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])

gaussian_fil_5x5 = 1/273*np.array([[1, 4, 7, 4, 1],
                                   [4, 16, 26, 16, 4],
                                   [7, 26, 41, 26, 7],
                                   [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]])


def space_filter(img, fil_type="CONV_RGB", fil=fil_sample, mode="SAME"):
    if mode == "SAME":
        # get the height and width of the convolution kernel
        h = fil.shape[0]
        w = fil.shape[1]

        # padding for the input image
        if h % 2 == 0:
            pad_h_top = h//2
            pad_h_bottom = h//2-1
        else:
            pad_h_top = pad_h_bottom = h//2

        if w % 2 == 0:
            pad_w_lef = h//2
            pad_w_rig = h//2-1
        else:
            pad_w_lef = pad_w_rig = h//2

        if fil_type == "GRAY":
            img = np.pad(img, ((pad_h_top, pad_h_bottom),
                               (pad_w_lef, pad_w_rig)), "constant")
        else:
            img = np.pad(img, ((pad_h_top, pad_h_bottom),
                               (pad_w_lef, pad_w_rig), (0, 0)), "constant")

    if fil_type == "CONV_RGB":
        conv_r = conv(img[:, :, 0], fil)
        conv_g = conv(img[:, :, 1], fil)
        conv_b = conv(img[:, :, 2], fil)
    elif fil_type == "MIDDLE_FILTER":
        conv_r = middle_filter(img[:, :, 0], fil)
        conv_g = middle_filter(img[:, :, 1], fil)
        conv_b = middle_filter(img[:, :, 2], fil)
    elif fil_type == "GRAY":
        conv_i = conv(img, fil)

    if fil_type == "GRAY":
        output_img = conv_i
    else:
        # combine the RGB channel
        output_img = np.dstack([conv_r, conv_g, conv_b])

    return output_img


def conv(img_1, fil):
    # get the height and width of the convolutional kernel
    fil_h = fil.shape[0]
    fil_w = fil.shape[1]

    # the output image shape
    conv_h = img_1.shape[0]-fil.shape[0]+1
    conv_w = img_1.shape[1]-fil.shape[1]+1

    # initialize the output image with zeros
    conv_output = np.zeros((conv_h, conv_w), dtype="uint8")

    for i in range(conv_h):
        for j in range(conv_w):
            conv_output[i][j] = weighted_sum(img_1[i:i+fil_w, j:j+fil_h], fil)

    return conv_output


def weighted_sum(img_e, fil):
    res = (img_e*fil).sum()
    if res < 0:
        res = 0
    elif res > 255:
        res = 255
    return res


def middle_filter(img_1, fil):
    # get the height and width of the filter kernel
    fil_h = fil.shape[0]
    fil_w = fil.shape[1]

    # the output image shape
    mid_h = img_1.shape[0]-fil.shape[0]+1
    mid_w = img_1.shape[1]-fil.shape[1]+1

    # initialize the output image with zeros
    mid_output = np.zeros((mid_h, mid_w), dtype="uint8")

    for i in range(mid_h):
        for j in range(mid_w):
            mid_output[i][j] = get_middle(img_1[i:i+fil_w, j:j+fil_h])

    return mid_output


def get_middle(img_e):
    e = img_e.flatten()
    res = np.median(e)
    return res


def main():
    img = cv2.imread("test2.jpg")   # read the image
    # change the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img)
    img_type = img.shape
    print(img_type)
    plt.imshow(img)  # show the image
    plt.show()

    fil = 1/9*np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])

    # mean filter
    # res = cv2.filter2D(img, -1, fil)  # use opencv
    # res = space_filter(img, "CONV_RGB", fil, "SAME")

    # middle filter
    # res = cv2.medianBlur(img, 3) # use opencv
    res = space_filter(img, "MIDDLE_FILTER", fil, "SAME")

    # Non-Local Means filter
    # res = cv2.fastNlMeansDenoising(img, None, 20.0, 5, 35)  # use opencv

    plt.imshow(res)
    plt.imsave("res.jpg", res)
    print(res.shape)
    plt.show()

    tar = cv2.imread("org2.jpg")
    tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
    print(psnr(tar, res, 512.0))  # calculate the PSNR


if __name__ == '__main__':
    main()
