import os
import cv2

input_Folder = "synthetic_030_images"
output_Folder = "processed_data"

if not os.path.exists(output_Folder):
    os.makedirs(output_Folder)

for file_Name in os.listdir(input_Folder):
    if file_Name.endswith(".tif"):
        image_Path=os.path.join(input_Folder, file_Name)
        image=cv2.imread(image_Path)

        # rgbden griye renk dönüşümü
        gray_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # median filtresi (tuz biber gürültüsünü temizlemek için)
        median = cv2.medianBlur(gray_Image, 5)
        # kontrast artırma
        # aydınlanma farklarını dengelemek için clahe yöntemi (kontrast artırma)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        contrast_Increased_Image = clahe.apply(median)
        # kontrast artırmadan dolayı oluşan tuz biber gürültüsü için tekrardan median filtresi
        median2 = cv2.medianBlur(contrast_Increased_Image, 51)
        # parlaklığı azaltmak için çıkarma
        substracted_Image = cv2.subtract(contrast_Increased_Image, median2)

        # işlenmiş resmi kaydetme
        output_Path = os.path.join(output_Folder, file_Name)
        cv2.imwrite(output_Path, substracted_Image)
        print(f"İşlendi ve kaydedildi: {file_Name}")

        # fotoğraflardan birini örnek olarak ekranda göstermek için
        if file_Name=="1GRAY.tif":
            cv2.imshow("Gri Goruntu", gray_Image)
            cv2.waitKey(0)
            cv2.imshow("median (5*5)", median)
            cv2.waitKey(0)
            cv2.imshow("kontrastli", contrast_Increased_Image)
            cv2.waitKey(0)
            cv2.imshow("cikarilmis", substracted_Image)
            cv2.waitKey(0)
