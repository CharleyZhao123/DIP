import numpy as np
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