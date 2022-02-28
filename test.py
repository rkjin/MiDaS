import cv2
import numpy as np

  
img = cv2.imread('freespace.jpg',cv2.IMREAD_GRAYSCALE)
# mask = np.full((480,640), 50)
# img[img > mask] = 0

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()