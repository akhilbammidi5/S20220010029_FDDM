import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_feature_matching(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        print("Error: Could not load one of the images.")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        ssd_distance1 = m.distance
        ssd_distance2 = n.distance

        if ssd_distance1 < 0.75 * ssd_distance2:
            good_matches.append(m)

    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches with SIFT')
    plt.axis('off')  
    plt.show()

    print("Number of good matches:", len(good_matches))
    if good_matches:
        print("Closest match SSD distance:", good_matches[0].distance)
        if len(good_matches) > 1:
            print("Second closest match SSD distance:", good_matches[1].distance)
            ratio_distance = good_matches[0].distance / good_matches[1].distance
            print("Ratio distance (closest/second closest):", ratio_distance)

image_path_1 = "img2.png"
image_path_2 = "img4.png"

sift_feature_matching(image_path_1, image_path_2)
