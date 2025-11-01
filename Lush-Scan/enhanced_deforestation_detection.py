import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('TkAgg')  # Set TkAgg backend
import matplotlib.pyplot as plt

# Load satellite images (RGB)
def load_images(file_paths, target_size=(256, 256)):
    images = []
    for path in file_paths:
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                images.append(img)
            else:
                print(f"Warning: Could not load image {path}")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return np.array(images)

# Gaussian blur
def apply_gaussian_blur(img, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

# Median filter
def apply_median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

# Adaptive thresholding only
def apply_thresholding(img, method='adaptive'):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

# Edge detection - Canny and Laplacian only
def detect_edges(img, method='canny'):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == 'canny':
        return cv2.Canny(gray, 50, 150)
    elif method == 'laplacian':
        return cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)
    else:
        return cv2.Canny(gray, 50, 150)

# Enhanced segmentation with HSV and LAB
def enhanced_segment_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_brown = np.array([0, 128, 128])
    upper_brown = np.array([255, 150, 150])
    brown_mask = cv2.inRange(lab, lower_brown, upper_brown)

    combined_mask = cv2.bitwise_or(green_mask, brown_mask)
    segmented_img = cv2.bitwise_and(img, img, mask=combined_mask)
    return segmented_img, combined_mask

# Feature extraction
def extract_enhanced_features(images):
    features_list = []
    for img in images:
        feature_vector = []

        # Filters
        blurred = apply_gaussian_blur(img)
        median_filtered = apply_median_filter(img)

        # Segmentation
        segmented_img, mask = enhanced_segment_image(img)

        # Edge detection
        canny_edges = detect_edges(img, 'canny')
        laplacian_edges = detect_edges(img, 'laplacian')

        # Thresholding
        adaptive_thresh = apply_thresholding(img)

        # Stats from original, blurred, segmented
        for processed_img in [img, blurred, segmented_img]:
            if len(processed_img.shape) == 3:
                for channel in range(3):
                    ch = processed_img[:, :, channel].flatten()
                    feature_vector.extend([
                        np.mean(ch), np.std(ch), np.median(ch),
                        np.percentile(ch, 25), np.percentile(ch, 75)
                    ])
            else:
                ch = processed_img.flatten()
                feature_vector.extend([
                    np.mean(ch), np.std(ch), np.median(ch),
                    np.percentile(ch, 25), np.percentile(ch, 75)
                ])

        # Texture
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray) ** 2
        homogeneity = 1.0 / (1.0 + contrast)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        energy = np.sum(hist ** 2)
        feature_vector.extend([contrast, homogeneity, energy])

        # Histograms
        hist_r = cv2.calcHist([img], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [32], [0, 256])
        feature_vector.extend(hist_r.flatten())
        feature_vector.extend(hist_g.flatten())
        feature_vector.extend(hist_b.flatten())

        # Vegetation ratio
        vegetation_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        feature_vector.append(vegetation_ratio)

        features_list.append(feature_vector)
    return np.array(features_list)

# Visualization
def visualize_processing_steps(img, img_index):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'Image Processing - Image {img_index + 1}', fontsize=16)

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(apply_gaussian_blur(img))
    axes[0, 1].set_title('Gaussian Blur')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(apply_median_filter(img))
    axes[0, 2].set_title('Median Filter')
    axes[0, 2].axis('off')

    segmented, mask = enhanced_segment_image(img)
    axes[1, 0].imshow(segmented)
    axes[1, 0].set_title('Segmented')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(detect_edges(img, 'canny'), cmap='gray')
    axes[1, 1].set_title('Canny Edges')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(apply_thresholding(img), cmap='gray')
    axes[1, 2].set_title('Adaptive Threshold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Label generation
def create_labels(num_samples, label_deforested_ratio=0.5):
    labels = np.zeros(num_samples)
    num_deforested = int(num_samples * label_deforested_ratio)
    labels[:num_deforested] = 1
    np.random.shuffle(labels)
    return labels

# Main
def main():
    print("Enhanced Deforestation Detection - Filtered & Segmented Only")
    file_paths = ["./img/image1.jpg", "./img/image2.jpeg", "./img/image3.jpg", "./img/image4.jpg"]
    images = load_images(file_paths)
    if len(images) == 0:
        print("No images loaded. Check file paths.")
        return

    print(f"{len(images)} images loaded.")
    print("Extracting features...")
    features = extract_enhanced_features(images)
    labels = create_labels(len(images))

    if len(images) > 2:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    else:
        X_train = X_test = features
        y_train = y_test = labels

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, target_names=['Non-Deforested', 'Deforested']))

    print("\nVisualizing image processing steps...")
    for i, img in enumerate(images):
        visualize_processing_steps(img, i)

if __name__ == "__main__":
    main()
