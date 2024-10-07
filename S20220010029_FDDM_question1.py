import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    dst = cv2.dilate(dst, None)
    
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    return img 

image_path_1 = "img2.png"
image_path_2 = "img4.png"

result_img1 = harris_corner_detection(image_path_1)
result_img2 = harris_corner_detection(image_path_2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners - Image 1')
plt.axis('off') 

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners - Image 2')
plt.axis('off') 
plt.tight_layout()
plt.show()
