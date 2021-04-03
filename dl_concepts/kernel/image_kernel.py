import numpy as np
import cv2
from matplotlib import pyplot as plt

# More image kernels can be found here:
# https://en.wikipedia.org/wiki/Kernel_(image_processing)
# Explained visually: https://setosa.io/ev/image-kernels/

sharpen_kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

laplacian_kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

blur_kernel = np.array([[0.0625, 0.125, 0.0625],
                      [0.125, 0.25, 0.125],
                      [0.0625, 0.125, 0.0625]])


def main():
    image = cv2.imread("figs/USF.jpeg")
    result = cv2.filter2D(image, -1, blur_kernel)
    # cv2.imshow('image',result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(result)
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()
