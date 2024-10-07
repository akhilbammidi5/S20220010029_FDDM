import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_feature_detection(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    return img_with_keypoints, descriptors 
image_path_1 = "img2.png"
image_path_2 = "img4.png"

result_img1, descriptors1 = sift_feature_detection(image_path_1)
result_img2, descriptors2 = sift_feature_detection(image_path_2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints - Image 1')
plt.axis('off') 
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints - Image 2')
plt.axis('off') 
plt.tight_layout()
plt.show()

print(f'Descriptors for Image 1: {descriptors1.shape if descriptors1 is not None else "No descriptors found"}')
print(f'Descriptors for Image 2: {descriptors2.shape if descriptors2 is not None else "No descriptors found"}')
