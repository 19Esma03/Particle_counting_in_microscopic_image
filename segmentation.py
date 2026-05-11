# =========================
# segmentation.py
# =========================

import cv2
import os
import numpy as np

def watershed_Segmentation(input_Folder, output_Folder):

    # output klasörü oluştur
    if not os.path.exists(output_Folder):
        os.makedirs(output_Folder)

    # klasördeki tüm tif dosyaları
    for file_Name in os.listdir(input_Folder):

        if file_Name.endswith(".tif"):

            image_Path = os.path.join(input_Folder, file_Name)

            # görüntüyü oku
            image = cv2.imread(image_Path)

            if image is None:
                print(f"Görüntü okunamadı: {file_Name}")
                continue

            # grayscale
            gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # OTSU threshold
            ret, thresh_Otsu = cv2.threshold(
                gray_Image,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # kernel
            kernel = np.ones((3,3), np.uint8)

            # Gürültü temizleme
            cleaned_Image = cv2.morphologyEx(
                thresh_Otsu,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1
            )

            # Hücre içlerini doldurma
            filled_Image = cv2.morphologyEx(
                cleaned_Image,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1
            )

            # Kesin arka plan
            sure_Background = cv2.dilate(
                filled_Image,
                kernel,
                iterations=3
            )

            # Distance transform
            dist_Transform = cv2.distanceTransform(
                filled_Image,
                cv2.DIST_L2,
                5
            )

            # Kesin ön plan
            ret, sure_Foreground = cv2.threshold(
                dist_Transform,
                0.5 * dist_Transform.max(),
                255,
                0
            )

            sure_Foreground = np.uint8(sure_Foreground)

            # Bilinmeyen bölge
            unknown = cv2.subtract(
                sure_Background,
                sure_Foreground
            )

            # Connected components
            ret, markers = cv2.connectedComponents(
                sure_Foreground
            )

            # Marker düzenleme
            markers = markers + 1
            markers[unknown == 255] = 0

            # Watershed için renkli görüntü
            colored_Image = cv2.cvtColor(
                gray_Image,
                cv2.COLOR_GRAY2BGR
            )

            # Watershed
            markers = cv2.watershed(
                colored_Image,
                markers
            )

            # Watershed sınırları kırmızı
            colored_Image[markers == -1] = [0, 0, 255]

            # output path
            output_Path = os.path.join(
                output_Folder,
                file_Name
            )

            # markerları kaydet (.npy)
            np.save(
                output_Path.replace(".tif", ".npy"),
                markers
            )

            # görüntüyü kaydet
            cv2.imwrite(
                output_Path,
                colored_Image
            )

            print(f"İşlendi: {file_Name}")

            # örnek görselleştirme
            if file_Name == "1GRAY.tif":

                cv2.imshow("Original", gray_Image)
                cv2.waitKey(0)

                cv2.imshow("OTSU", thresh_Otsu)
                cv2.waitKey(0)

                cv2.imshow("Sure Foreground", sure_Foreground)
                cv2.waitKey(0)

                cv2.imshow("Unknown Region", unknown)
                cv2.waitKey(0)

                cv2.imshow("Watershed Result", colored_Image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()