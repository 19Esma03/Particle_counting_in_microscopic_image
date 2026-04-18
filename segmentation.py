import cv2
import os
import numpy as np

input_Folder = "processed_data"
output_Folder = "segmented_data"

if not os.path.exists(output_Folder):
    os.makedirs(output_Folder)

for file_Name in os.listdir(input_Folder):
    if file_Name.endswith(".tif"):
        image_Path=os.path.join(input_Folder, file_Name)

        image=cv2.imread(image_Path)
        gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh_Otsu = cv2.threshold(gray_Image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # kernel tanımlama
        kernel=np.ones((3,3), np.uint8)
        # gürültü temizleme, çok küçük beyaz pikselleri temizler
        cleaned_Image=cv2.morphologyEx(thresh_Otsu, cv2.MORPH_OPEN, kernel, iterations=1)
        # hücre doldurma, hücrelerin içinde küçük siyahlıklar varsa oraları doldurur
        filled_Image=cv2.morphologyEx(cleaned_Image, cv2.MORPH_CLOSE, kernel, iterations=1)

        # kesin arka plan, boşlukları bulmak için
        sure_Background = cv2.dilate(filled_Image, kernel, iterations=3)

        # kesin ön plan, birbirine değen hücrelerin merkezlerini bulmak için
        dist_Transform = cv2.distanceTransform(filled_Image, cv2.DIST_L2, 5)
        ret, sure_Foreground = cv2.threshold(dist_Transform, 0.7 * dist_Transform.max(), 255, 0)

        # bilinmeyen bölge, kesin arka plan ile kesin ön plan arasındaki bölgeye göre sınırı belirler
        sure_Foreground = np.uint8(sure_Foreground)
        unknown=cv2.subtract(sure_Background,sure_Foreground)

        # her hücre merkezine etiket verme
        ret, markers=cv2.connectedComponents(sure_Foreground)
        # arka planı 1, bilinmeyen bölgeyi 0 yapma
        markers=markers+1
        markers[unknown == 255] = 0

        # watershed algoritması
        colored_Image=cv2.cvtColor(gray_Image, cv2.COLOR_GRAY2BGR) # watershed için fotoğraf renkli yapıldı
        markers=cv2.watershed(colored_Image, markers)

        output_Path = os.path.join(output_Folder, file_Name)
        cv2.imwrite(output_Path, colored_Image)
        print(f"İşlendi ve kaydedildi: {file_Name}")

        # bir fotoğrafta bütün yöntemleri uygulayıp örnek olarak ekranda göstermek için
        if file_Name == "1GRAY.tif":
            cv2.imshow("otsu", thresh_Otsu)
            cv2.waitKey(0)
            thresh_Mean = cv2.adaptiveThreshold(gray_Image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imshow("adaptive mean", thresh_Mean)
            cv2.waitKey(0)
            thresh_Gaussian = cv2.adaptiveThreshold(gray_Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                    11, 2)
            cv2.imshow("adaptive gaussian", thresh_Gaussian)
            cv2.waitKey(0)

            cv2.imshow("sure background", sure_Background)
            cv2.waitKey(0)
            cv2.imshow("sure foreground", sure_Foreground)
            cv2.waitKey(0)
            cv2.imshow("unknown", unknown)
            cv2.waitKey(0)

            colored_Image[markers == -1] = [0, 0, 255]  # kırmızı sınır çizmek için
            cv2.imshow("watershed", colored_Image)
            cv2.waitKey(0)