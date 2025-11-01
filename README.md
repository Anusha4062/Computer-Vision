# 🧠 Computer Vision Project Collection

This repository contains multiple **Computer Vision mini-projects** implemented using **Python** and **OpenCV**.
Each project demonstrates real-world image processing and object detection concepts such as facial recognition, object detection using YOLO, color segmentation for crop analysis, satellite-based deforestation detection, and more.

---

## 📁 Repository Structure

| Folder / File                              | Description                                                                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `Face and Eye Detection using OpenCV/`     | Detects human faces and eyes in real-time using Haar Cascade Classifiers.                           |
| `Object Detection and Recognition/`        | Performs real-time object detection using YOLOv4 trained on the COCO dataset.                       |
| `Crop Health Monitoring/`                  | Analyzes plant leaf images and detects unhealthy regions using color segmentation.                  |
| `License Plate Detection and Recognition/` | Detects vehicle license plates and extracts text using OCR (Tesseract).                             |
| `Lush Scan (Deforestation Detection)/`     | Detects deforested regions in satellite images using segmentation and Random Forest classification. |
| `.gitignore`                               | Specifies ignored files such as large model weights or data files.                                  |
| `README.md`                                | Project documentation file (this file).                                                             |

---

## 🚀 Projects Overview

### 1️⃣ Face and Eye Detection using OpenCV

**Goal:** Detect human faces and eyes in live camera feed or static images.

**Technique Used:**

| Feature | Description                                                                    |
| ------- | ------------------------------------------------------------------------------ |
| Library | OpenCV (Haar Cascades)                                                         |
| Model   | `haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`                   |
| Input   | Webcam or image                                                                |
| Output  | Image/video frame with detected faces and eyes highlighted with bounding boxes |

---

### 2️⃣ Object Detection and Recognition

**Goal:** Detect and recognize multiple objects in an image or real-time video stream.

**Technique Used:**

| Feature   | Description                                    |
| --------- | ---------------------------------------------- |
| Model     | YOLOv4 (You Only Look Once)                    |
| Framework | OpenCV DNN Module                              |
| Dataset   | COCO (Common Objects in Context)               |
| Input     | Camera feed or image                           |
| Output    | Labeled bounding boxes around detected objects |

**Model Files Used:**

* `yolov4.weights`
* `yolov4.cfg`
* `coco.names`

---

### 3️⃣ Crop Health Monitoring using Color Segmentation

**Goal:** Analyze leaf images to determine crop health by segmenting green and yellow areas.

**Technique Used:**

| Feature | Description                                                                                     |
| ------- | ----------------------------------------------------------------------------------------------- |
| Model   | Custom HSV color segmentation (no pre-trained model)                                            |
| Library | OpenCV, NumPy, Matplotlib                                                                       |
| Input   | Leaf image                                                                                      |
| Output  | Percentage of healthy (green) and unhealthy (yellow/brown) regions with bar graph visualization |

**Status Classification:**

| Health % | Condition   |
| -------- | ----------- |
| > 70%    | ✅ Healthy   |
| 40–70%   | ⚠️ Moderate |
| < 40%    | ❌ Unhealthy |

---

### 4️⃣ License Plate Detection and Recognition

**Goal:** Identify and extract vehicle license plate numbers from images.

**Technique Used:**

| Feature | Description                                            |
| ------- | ------------------------------------------------------ |
| Model   | Haar Cascade for plate detection                       |
| OCR     | Tesseract for character recognition                    |
| Input   | Car image or video feed                                |
| Output  | Extracted license plate region + recognized plate text |

---

### 5️⃣ Lush Scan – Deforestation Detection from Satellite Images

**Goal:** Detect and classify deforested regions from satellite images using color-based segmentation and machine learning.

**Technique Used:**

| Feature            | Description                                                    |
| ------------------ | -------------------------------------------------------------- |
| Algorithm          | Random Forest Classifier                                       |
| Feature Extraction | Color histogram + segmented pixel intensities                  |
| Segmentation       | Color-based masking (using OpenCV `inRange` method)            |
| Libraries          | OpenCV, NumPy, Matplotlib, scikit-learn                        |
| Input              | Satellite images (RGB format)                                  |
| Output             | Segmented deforested area visualization + model accuracy score |

**Workflow:**

1. Load and resize multiple satellite images.
2. Apply color segmentation to isolate forest and non-forest areas.
3. Extract pixel and histogram features for each image.
4. Train a **Random Forest Classifier** to predict deforested areas.
5. Display side-by-side visualization of original and segmented images.

**Example Output:**

* Accuracy of trained model displayed in console.
* Segmented images showing deforested regions.
* Visualization using Matplotlib.

---

## 🧰 Technologies Used

| Technology        | Purpose                                                 |
| ----------------- | ------------------------------------------------------- |
| **Python**        | Core programming language                               |
| **OpenCV**        | Image processing, detection, segmentation               |
| **NumPy**         | Array operations and mathematical computations          |
| **Matplotlib**    | Visualization and result plotting                       |
| **scikit-learn**  | Model training and evaluation (Random Forest)           |
| **Tesseract OCR** | Optical character recognition for license plates        |
| **YOLOv4**        | Object detection model for recognizing multiple objects |

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AnouskaJ/computer_vision_project.git
   cd computer_vision_project
   ```

2. **Install Dependencies**

   ```bash
   pip install opencv-python numpy matplotlib scikit-learn pytesseract
   ```

3. **Run Individual Projects**

   ```bash
   python face_eye_detection.py
   python object_detection.py
   python crop_health_monitor.py
   python license_plate_detection.py
   python lush_scan_deforestation.py
   ```

4. **Add your own images/videos**

   * Place input files in the corresponding project folders.
   * Modify the `file_paths` list in the script as needed.

---

## 📊 Example Outputs

| Project          | Output Example                                                 |
| ---------------- | -------------------------------------------------------------- |
| Face Detection   | Bounding boxes around detected faces                           |
| Object Detection | Multiple objects labeled with confidence                       |
| Crop Health      | Green and yellow region comparison chart                       |
| License Plate    | Recognized plate text displayed                                |
| Lush Scan        | Segmented forest and deforested visualization + accuracy score |
