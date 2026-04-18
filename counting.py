import cv2
import numpy as np
import glob

file_path ="segmented_data/*.npy"

def counter():
    for file in glob.glob(file_path):

        markers = np.load(file)  

        unique_labels = np.unique(markers)
        count = 0 

        image = cv2.imread(file.replace(".npy", ".tif"))

        for label in unique_labels:
            if label <= 1:
                continue

            mask = np.uint8(markers == label)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = contours[0]
            area = cv2.contourArea(contour)

            if area < 150:  
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if perimeter == 0:
                continue

            if circularity < 0.5:
                continue

            count += 1
            cv2.drawContours(image, [contour], -1, (0,255,0), 2)


        cv2.imshow("Cells", image)
        cv2.waitKey(0)