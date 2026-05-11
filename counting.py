# =========================
# counting.py
# =========================

import numpy as np
import glob

def counter():

    file_path = "segmented_data/*.npy"

    total_Count = 0

    for file in glob.glob(file_path):

        # marker yükle
        markers = np.load(file)

        # benzersiz label'lar
        labels = np.unique(markers)

        # hücre label'ları
        # -1 = boundary
        # 1 = background
        cell_Labels = labels[labels > 1]

        # hücre sayısı
        count = len(cell_Labels)

        total_Count += count

        print(f"{file} -> Cell Count: {count}")

    print(f"Toplam Hücre Sayısı: {total_Count}")
