import cv2
import numpy as np
from PIL import Image

# image = cv2.imread("data/254p_Thermal_Images/53448.jpg")
imagergb = cv2.imread("data/254p_RGB_Images/14595.jpg")
imageir = Image.open("data/254p_Thermal_Images/14595.jpg")
image = cv2.cvtColor(np.array(imageir), cv2.COLOR_RGB2BGR)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower_orange = np.array([2, 75, 75])  # Lower HSV values for orange
# upper_orange = np.array([22, 255, 255])  # Upper HSV values for orange

# mask = cv2.inRange(hsv, lower_orange, upper_orange)
lower_red = np.array([0, 0, 0])    # Lower bound for red color
upper_red = np.array([60, 255, 255])   # Upper bound for red color
mask = cv2.inRange(hsv, lower_red, upper_red)
mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1] 

output_image = np.zeros_like(image)
output_image[mask_binary == 255] = [255, 255, 255]
print("Shape of mask_binary:", output_image.shape)
single_channel_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
print("Shape of mask_binary:", single_channel_gray.shape)

# mask_single_channel = cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(single_channel_gray, connectivity=8)

print("Number of areas:", num_labels - 1) 
labeled_image = np.zeros_like(labels, dtype=np.uint8)
total_area=0
area_count=0
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    print("Area of component", i, ":", area)
    if area > 50:
        labeled_image[labels == i] =255
        total_area+=area
        area_count+=1
print("Number of areas under fire",area_count)
print("Total area under fire",total_area)
cv2.imshow("RGB Image",imagergb)
cv2.imshow("IR Image", image)
cv2.imshow("Output Image", output_image)
cv2.imshow('Labeled Image', labeled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()




