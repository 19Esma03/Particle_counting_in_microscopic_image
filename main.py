
import cv2
import os
import numpy as np

input_Folder = "processed_data"
output_Folder = "segmented_data"


from segmentation import watershed_Segmentation
from counting import counter
from evalution import evaluate

watershed_Segmentation(input_Folder, output_Folder)

counter()
evaluate()



