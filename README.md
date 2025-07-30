# 🧠 Brain Tumor Detection Using CNN

This project presents a deep learning model built with **Convolutional Neural Networks (CNN)** to classify brain MRI scans into tumor and non-tumor categories. The model is trained and evaluated using a publicly available dataset and showcases the use of Python, TensorFlow, and OpenCV.

---

## 🎯 Objective

To automate the detection of brain tumors from MRI images using CNN architecture, assisting early diagnosis through AI-based medical image classification.

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| Python         | Programming language |
| TensorFlow / Keras | Deep Learning framework |
| OpenCV         | Image preprocessing |
| NumPy / Pandas | Data manipulation |
| Matplotlib     | Visualization |

---

## 🧪 Workflow Overview

1. 📁 Load and preprocess the dataset (resizing, grayscale, normalization)
2. 🧠 Build a CNN model using TensorFlow/Keras
3. 🔁 Train and validate on labeled brain MRI scans
4. 📊 Evaluate accuracy, precision, and recall
5. 🖼️ Predict and visualize test MRI results

---

## 🧠 Model Architecture

- 3 Convolutional Layers
- ReLU activation
- MaxPooling
- Flatten → Dense Layers
- Output Layer with Softmax

> Model achieved ~95% accuracy on validation data.

---

## 📂 Project Structure

```
cnn-brain-tumor-detection/
├── dataset/
│   ├── yes/ (tumor images)
│   └── no/  (non-tumor images)
├── model/
│   └── tumor_model.h5
├── src/
│   └── cnn_model.py
├── notebooks/
│   └── tumor_detection.ipynb
├── results/
│   └── predictions.png
└── README.md
```

---

## 📸 Results

> Add graphs of training vs validation accuracy/loss, sample test predictions.

---

## 📚 Dataset Source

- [Kaggle: Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## ✅ Conclusion

This project demonstrates the effectiveness of deep learning in medical image classification, especially for early-stage tumor detection. It can be enhanced further with transfer learning and dataset expansion for real-world deployment.

---

## 🔐 Ethical Note

This model is for **educational and research purposes only**. Medical diagnosis must be performed by certified professionals.

