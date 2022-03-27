# Rodrigo Zimmermann
from matplotlib import pyplot as plt
import cv2
import numpy as np

image = cv2.imread('mama6.png',0)

kernel = np.ones((11,11),np.uint8)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
resultOne = cv2.absdiff(image, opening)

kernel = np.ones((11,11),np.uint8)
closing = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
resultTwo = cv2.absdiff(closing, image)

result = cv2.add(image, resultOne)
finalResult = cv2.absdiff(result, resultTwo)
ret,thresholding = cv2.threshold(finalResult,100,255,cv2.THRESH_BINARY)
plt.imshow(thresholding, 'gray')
plt.title('Imagem final')
plt.show()