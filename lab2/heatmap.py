import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


files_list = []
for path, subdirs, files in os.walk("/home/andrei/Downloads/archive"):
    for name in files:
        files_list.append(str(os.path.join(path, name)))


for color in [0, 1, 2]:
    result_img = np.zeros(shape=cv2.imread(files_list[0], 0).shape)

    for image in files_list:
        img = cv2.imread(image)
        if img is not None:
            rows, cols, _ = img.shape

            for i in range(rows):
                for j in range(cols):
                    result_img[i, j] += img[i, j][color]

    plt.imshow(result_img, cmap='hot', interpolation='nearest')
    plt.show()

#        f8666878-6995-4e12-93b9-c20b02b606ba___Keller.St_CG 1985.JPG  ... 5caecfc6-8e1d-401b-8014-763dec01cb98___RS_Early.B 8063.JPG
# count                                                   1            ...                                                  1
# unique                                                  1            ...                                                  1
# top     [[[125, 104, 113], [134, 113, 122], [116, 95, ...            ...  [[[152, 145, 150], [143, 136, 141], [157, 150,...
# freq                                                    1            ...                                                  1
#
# [4 rows x 20639 columns]

