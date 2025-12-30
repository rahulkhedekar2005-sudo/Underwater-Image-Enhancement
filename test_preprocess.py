import cv2
from utils.preprocessing import preprocess

# CHANGE image name if needed
img = cv2.imread("dataset/train/input/2_img_.png")


enhanced = preprocess(img)

cv2.imshow("Original Image", img)
cv2.imshow("Enhanced Image", enhanced)

cv2.waitKey(0)
cv2.destroyAllWindows()
