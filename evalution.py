import numpy as np
import glob
import cv2


SEGMENTED_FOLDER = "segmented_data"
# Ortalama hücre alanının kaç standart sapma dışında kalanlar anormal sayılsın
ANOMALY_THRESHOLD = 2.5


def analyze_image(npy_path):
    markers = np.load(npy_path)
    unique_labels = np.unique(markers)
    cell_labels = unique_labels[unique_labels > 1]
    cell_count = len(cell_labels)
    areas = [np.sum(markers == label) for label in cell_labels]
    return cell_count, cell_labels, areas  # ← cell_labels eklendi


def find_anomalies(cell_labels,areas, threshold=ANOMALY_THRESHOLD):
    if len(areas) < 2:
        return []

    mean = np.mean(areas)
    std = np.std(areas)

    anomalies = []
    for label, area in zip(cell_labels, areas):
        if std > 0 and abs(area - mean) > threshold * std:
            if area < mean:
                reason = "çok küçük (gürültü olabilir)"
            else:
                reason = "çok büyük (birleşik hücre olabilir)"
            anomalies.append((label, area, reason))

    return anomalies


def visualize(npy_path):
    markers = np.load(npy_path)
    tif_path = npy_path.replace(".npy", ".tif")
    image = cv2.imread(tif_path)

    if image is None:
        print(f"  ⚠️  Görüntü okunamadı: {tif_path}")
        return

    unique_labels = np.unique(markers)
    cell_labels = unique_labels[unique_labels > 1]

    areas = [np.sum(markers == label) for label in cell_labels]

    if len(areas) >= 2:
        mean = np.mean(areas)
        std = np.std(areas)
    else:
        mean, std = 0, 0

    for label, area in zip(cell_labels, areas):
        mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Anormal segment → kırmızı, normal → yeşil
        if std > 0 and abs(area - mean) > ANOMALY_THRESHOLD * std:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.drawContours(image, contours, -1, color, 2)

    file_name = npy_path.split("/")[-1].replace(".npy", ".tif")
    cv2.imshow(f"{file_name}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def evaluate():
    files = sorted(glob.glob(f"{SEGMENTED_FOLDER}/*.npy"))

    if not files:
        print(f"'{SEGMENTED_FOLDER}' klasöründe .npy dosyası bulunamadı.")
        return

    all_counts = []
    total_anomalies = 0

    print("=" * 55)
    print(f"{'DOSYA':<20} {'HÜCRE':>6} {'ORT.ALAN':>10} {'STD':>8} {'ANORMAL':>8}")
    print("=" * 55)

    for npy_path in files:
        file_name = npy_path.split("/")[-1].replace(".npy", ".tif")
        cell_count, cell_labels, areas = analyze_image(npy_path)
        anomalies = find_anomalies(cell_labels, areas)

        all_counts.append(cell_count)
        total_anomalies += len(anomalies)

        mean_area = np.mean(areas) if areas else 0
        std_area = np.std(areas) if areas else 0
        anomaly_flag = f"⚠️  {len(anomalies)}" if anomalies else "✓  0"

        print(f"{file_name:<20} {cell_count:>6} {mean_area:>10.1f} {std_area:>8.1f} {anomaly_flag:>8}")

        for label, area, reason in anomalies:
            print(f"    └─ Label {label}: {area}px² → {reason}")

    print("=" * 55)

    if len(all_counts) > 1:
        count_mean = np.mean(all_counts)
        count_std = np.std(all_counts)
        consistency = "tutarlı ✓" if count_std < count_mean * 0.3 else "tutarsız "

        print(f"\n GENEL ÖZET")
        print(f"  Toplam görüntü        : {len(files)}")
        print(f"  Ortalama hücre sayısı : {count_mean:.1f}")
        print(f"  Hücre sayısı std      : {count_std:.1f}  ({consistency})")
        print(f"  Toplam anormal segment: {total_anomalies}")

    print("\n🔍 Görsel kontrol başlıyor... (her görüntü için bir tuşa basın)")
    for npy_path in files:
        visualize(npy_path)


if __name__ == "__main__":
    evaluate()