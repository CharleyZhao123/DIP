import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test3.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("img", img)

plt.hist(img.ravel(), 256)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()