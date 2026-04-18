import cv2
import os
import numpy as np

def watershed_Segmentation(input_Folder, output_Folder):

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
            cleaned_Image=cv2.morphologyEx(thresh_Otsu, cv2.MORPH_OPEN, kernel, iterations=2)
            # hücre doldurma, hücrelerin içinde küçük siyahlıklar varsa oraları doldurur
            filled_Image=cv2.morphologyEx(cleaned_Image, cv2.MORPH_CLOSE, kernel, iterations=1)

            # kesin arka plan, boşlukları bulmak için
            sure_Background = cv2.dilate(filled_Image, kernel, iterations=3)

            # kesin ön plan, birbirine değen hücrelerin merkezlerini bulmak için
            dist_Transform = cv2.distanceTransform(filled_Image, cv2.DIST_L2, 5)
            #Normalize
            cv2.normalize(dist_Transform, dist_Transform, 0, 1.0, cv2.NORM_MINMAX)
            
            ret, sure_Foreground = cv2.threshold(dist_Transform, 0.3, 1.0, cv2.THRESH_BINARY)
            # bilinmeyen bölge, kesin arka plan ile kesin ön plan arasındaki bölgeye göre sınırı belirler
            sure_Foreground = np.uint8(sure_Foreground * 255)

            sure_Foreground = cv2.erode(sure_Foreground, kernel, iterations=1)

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
            colored_Image[markers == -1] = [0, 0, 255]  # kırmızı sınır çizmek için

            np.save(output_Path.replace(".tif", ".npy"), markers)

            cv2.imwrite(output_Path, colored_Image)
            print(f"Processed and saved: {file_Name}")

            # bir fotoğrafta bütün yöntemleri uygulayıp örnek olarak ekranda göstermek için