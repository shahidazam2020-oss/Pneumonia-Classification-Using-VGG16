<p align="center">

 <a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16/stargazers">
    <img src="https://img.shields.io/github/stars/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16?style=social&cacheSeconds=60" alt="GitHub Stars">
  </a>
 <a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16/forks">
    <img src="https://img.shields.io/github/forks/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16?style=social&cacheSeconds=60" alt="GitHub Forks">
  </a>
<a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16/commits">
    <img src="https://img.shields.io/github/last-commit/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16?display_timestamp=committer&style=flat-square" alt="Last Commit">
  </a>

  </p>

---
<div align="center">

# 🫁 Pneumonia Classification Using VGG16

### Deep Learning • Transfer Learning • Medical Image Classification

<p>
  <img src="https://img.shields.io/badge/Deep%20Learning-VGG16-blue?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Medical%20AI-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge&logo=python&logoColor=white" />
</p>

<br>

<a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16">
  <img src="https://img.shields.io/badge/📂_EXPLORE-GitHub_Repository-181717?style=for-the-badge&logo=github" />
</a>

<a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16/issues">
  <img src="https://img.shields.io/badge/🐛_REPORT-Open_an_Issue-red?style=for-the-badge&logo=github" />
</a>

<a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16">
  <img src="https://img.shields.io/badge/⭐_SUPPORT-Star_Repository-yellow?style=for-the-badge&logo=github" />
</a>

<br><br>

**An AI-powered image classification project leveraging VGG16 transfer learning to distinguish between Normal and Pneumonia chest X-ray images.**

<br>

[Overview](#-project-overview) •
[Architecture](#-model-architecture) •
[Dataset](#-dataset-overview) •
[Methodology](#-methodology) •
[Results](#-model-evaluation--results) •
[Installation](#-installation--execution)

</div>

---

# 📌 Project Overview

Pneumonia is a respiratory infection that affects the lungs and can cause serious health complications.

This project explores the application of **Deep Learning and Transfer Learning** for classifying chest X-ray images into two categories:

- 🟢 **Normal:** Chest X-ray images representing the Normal class.
- 🔴 **Pneumonia:** Chest X-ray images representing the Pneumonia class.

The project utilizes the VGG16 convolutional neural network pretrained on ImageNet as a feature extraction backbone.

A custom classification head is added to the pretrained architecture to perform binary image classification.

The implementation covers image preprocessing, data augmentation, model development, training, validation, performance evaluation, and visualization.

> **Primary Objective:** Investigate the effectiveness of VGG16 transfer learning for automated chest X-ray image classification.

---

# 🎯 Project Objectives

The main objectives of this project are:

- 🫁 Develop a deep learning framework for pneumonia image classification.
- 🧠 Implement transfer learning using the pretrained VGG16 architecture.
- 🖼️ Prepare chest X-ray images for deep learning.
- 🔄 Apply data augmentation to improve model generalization.
- ⚙️ Develop a custom binary classification head.
- 📈 Monitor training and validation performance.
- 📊 Evaluate classification performance using multiple metrics.
- 🔍 Analyze model predictions through a confusion matrix.
- 📋 Generate a detailed classification report.

---

# 📊 Project Dashboard

<div align="center">

| Project Attribute | Details |
|:---|:---|
| 🫁 Project | Pneumonia Classification |
| 🧠 Architecture | VGG16 |
| 🔬 Learning Approach | Transfer Learning |
| 🖼️ Input Type | Chest X-ray Images |
| 🎯 Classification | Binary Classification |
| 🟢 Class 1 | Normal |
| 🔴 Class 2 | Pneumonia |
| 📐 Input Dimensions | 224 × 224 × 3 |
| ⚙️ Optimizer | Adam |
| 📉 Loss Function | Binary Cross-Entropy |
| 🔄 Training Epochs | 30 |
| 📦 Batch Size | 32 |
| 💻 Framework | TensorFlow / Keras |

</div>

---

# 📂 Dataset Overview

The project uses the **Chest X-Ray Images (Pneumonia)** dataset.

The dataset contains chest X-ray images categorized into two classes:

| Category | Description |
|:---|:---|
| 🟢 NORMAL | Images belonging to the Normal category |
| 🔴 PNEUMONIA | Images belonging to the Pneumonia category |

### 📁 Dataset Organization

The dataset is organized into three main subsets:

```text
Chest_X-Ray_Dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 📐 Image Specifications

| Parameter | Configuration |
|:---|:---|
| Image Type | Chest X-ray |
| Image Format | JPEG |
| Input Width | 224 Pixels |
| Input Height | 224 Pixels |
| Color Channels | 3 |
| Model Input | 224 × 224 × 3 |
| Classification Classes | 2 |

### 🔗 Dataset Source

The dataset can be accessed through Kaggle:

<a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">
  <img src="https://img.shields.io/badge/Kaggle-Access_Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
</a>

---

# 🔬 Methodology

The project follows a structured deep learning workflow.

### 🧭 Complete Project Workflow

```text
             🖼️ CHEST X-RAY DATASET
                       |
                       ▼
               📂 DATA LOADING
                       |
                       ▼
              🧹 IMAGE PREPROCESSING
                       |
                       ▼
              🔄 DATA AUGMENTATION
                       |
                       ▼
             🧠 PRETRAINED VGG16
                       |
                       ▼
              ⚙️ CUSTOM CLASSIFIER
                       |
                       ▼
                🤖 MODEL TRAINING
                       |
                       ▼
              📈 MODEL VALIDATION
                       |
                       ▼
               🧪 MODEL TESTING
                       |
                       ▼
             📊 PERFORMANCE ANALYSIS
                       |
                       ▼
              🏆 FINAL PREDICTIONS
```

---

# 🧹 Image Preprocessing

Image preprocessing is an important stage in preparing medical images for deep learning.

The project applies the following preprocessing operations:

### 1. 📐 Image Resizing

All images are resized to:

```python
224 × 224 × 3
```

This input dimension is compatible with the VGG16 architecture.

### 2. 🔢 Image Normalization

Pixel values are normalized using:

```python
rescale = 1./255
```

This converts pixel values from the original range of 0–255 to a normalized range of 0–1.

### 3. 🔄 Data Augmentation

Data augmentation is applied to the training dataset to introduce image variations.

The implemented augmentation techniques include:

| Technique | Purpose |
|:---|:---|
| 🔄 Rotation | Introduces rotational variations |
| 🔍 Zoom | Introduces scale variations |
| ↔️ Horizontal Flip | Introduces horizontal image variations |
| 📊 Rescaling | Normalizes pixel values |

Validation and testing images are rescaled without applying training augmentation.

---

# 🧠 Model Architecture

## VGG16 Transfer Learning

VGG16 is a deep convolutional neural network developed for image recognition tasks.

In this project, the pretrained VGG16 architecture is used as a feature extractor.

The original classification head is removed, and custom dense layers are introduced for binary classification.

### ⚙️ Pretrained Model Configuration

```python
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False
```

The pretrained convolutional layers are frozen during training.

### 🏗️ Custom Classification Architecture

```text
          INPUT IMAGE
        224 × 224 × 3
                |
                ▼
       PRETRAINED VGG16
        FEATURE EXTRACTOR
                |
                ▼
          FLATTEN LAYER
                |
                ▼
       DENSE LAYER (512)
           ReLU
                |
                ▼
        DROPOUT LAYER
             0.5
                |
                ▼
       OUTPUT LAYER (1)
            Sigmoid
                |
                ▼
       BINARY CLASSIFICATION
                |
          ┌─────┴─────┐
          ▼           ▼
        NORMAL     PNEUMONIA
```

### 📋 Model Configuration

| Parameter | Value |
|:---|:---|
| Base Architecture | VGG16 |
| Pretrained Weights | ImageNet |
| Include Top | False |
| Input Shape | 224 × 224 × 3 |
| Feature Extractor | Frozen VGG16 |
| Dense Neurons | 512 |
| Hidden Activation | ReLU |
| Dropout | 0.5 |
| Output Neurons | 1 |
| Output Activation | Sigmoid |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Cross-Entropy |
| Training Epochs | 30 |
| Batch Size | 32 |

---

# ⚙️ Model Training

The VGG16-based classification model is trained using the prepared chest X-ray training dataset.

Training and validation performance are monitored throughout the learning process.

### 🎯 Training Configuration

- Pretrained ImageNet weights.
- Frozen VGG16 feature extraction layers.
- Custom dense classification head.
- Adam optimizer.
- Binary cross-entropy loss.
- Batch size of 32.
- Maximum of 30 training epochs.

### 📈 Training Monitoring

Two important performance indicators are monitored:

**Training and Validation Loss**

Evaluates how the model's prediction error changes throughout training.

**Training and Validation Accuracy**

Compares classification accuracy across training and validation datasets.

---

# 📊 Model Evaluation & Results

After training, the model is evaluated using the independent test dataset.

The evaluation process includes:

| Metric | Purpose |
|:---|:---|
| 🎯 Test Accuracy | Overall classification correctness |
| 📉 Test Loss | Prediction error |
| 🔍 Precision | Reliability of positive predictions |
| ❤️ Recall | Identification of actual positive cases |
| ⚖️ F1-Score | Balance between precision and recall |
| 🧩 Confusion Matrix | Classification error analysis |

### 🏆 Performance Results

| Evaluation Metric | Result |
|:---|:---|
| Test Accuracy | Add actual result |
| Test Loss | Add actual result |
| Normal Precision | Add actual result |
| Normal Recall | Add actual result |
| Pneumonia Precision | Add actual result |
| Pneumonia Recall | Add actual result |
| Overall F1-Score | Add actual result |

> **Note:** Replace the placeholders with actual results generated by your notebook. No unverified accuracy or performance claims are reported here.

---

# 🔥 Confusion Matrix

A confusion matrix is used to examine the classification performance of the model.

It provides information about correctly and incorrectly classified chest X-ray images.

### 🧩 Confusion Matrix Structure

| Actual / Predicted | Normal | Pneumonia |
|:---|:---:|:---:|
| Normal | True Negative (TN) | False Positive (FP) |
| Pneumonia | False Negative (FN) | True Positive (TP) |

### 📌 Interpretation

- **True Negative:** Normal images correctly classified as Normal.
- **True Positive:** Pneumonia images correctly classified as Pneumonia.
- **False Positive:** Normal images incorrectly classified as Pneumonia.
- **False Negative:** Pneumonia images incorrectly classified as Normal.

False negatives are particularly important to examine in medical image classification because they represent pneumonia cases incorrectly classified as Normal.

---

# 📈 Learning Curves & Visualization

The project generates visualizations to examine model training and classification performance.

### 1. 📉 Training and Validation Loss

The loss visualization helps evaluate how prediction errors change throughout training.

### 2. 📈 Training and Validation Accuracy

The accuracy visualization provides insight into the model's learning behavior.

### 3. 🔥 Confusion Matrix

The confusion matrix helps identify classification strengths and errors across both categories.

### 4. 🖼️ Sample Chest X-ray Images

Sample images can be used to inspect the dataset and verify image labeling.

> **Visualization tip:** Add your actual generated plots to a `results/` directory and embed them below when available.

Example:

```markdown
![Training and Validation Loss](results/training_loss.png)

![Training and Validation Accuracy](results/training_accuracy.png)

![Confusion Matrix](results/confusion_matrix.png)

![Sample Chest X-ray Images](results/sample_images.png)
```

---

# 🚀 Installation & Execution

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16.git
```

### 2. Navigate to the Project Directory

```bash
cd Pneumonia-Classification-Using-VGG16
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas jupyter
```

### 4. Download the Dataset

Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

<a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">
  <img src="https://img.shields.io/badge/DOWNLOAD-Kaggle_Dataset-20BEFF?style=for-the-badge&logo=kaggle" />
</a>

Organize the dataset into the training, validation, and testing directories described above.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 6. Execute the Notebook

Open:

```text
Pneumonia Classiifcation.ipynb
```

Execute the notebook cells sequentially to perform preprocessing, model training, validation, and evaluation.

---

# 🛠️ Technologies & Tools

The project utilizes the following technologies:

<div align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge" />
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge" />
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
<img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />

</div>

---

# 📂 Repository Structure

```text
Pneumonia-Classification-Using-VGG16/
│
├── 📓 Pneumonia Classiifcation.ipynb
│
└── 📘 README.md
```

The notebook contains the data preprocessing, augmentation, VGG16 model implementation, training, evaluation, and visualization workflow.

---

# 🧠 Research Limitations

Although this project demonstrates the application of transfer learning for medical image classification, several limitations should be considered.

- **Dataset Dependency:** Model performance depends on the quality and distribution of the selected dataset.
- **Generalizability:** Performance on external datasets and different patient populations requires independent evaluation.
- **Model Interpretability:** VGG16 predictions require additional explainability techniques to better understand the visual features influencing classification.
- **Clinical Validation:** The model has not been established as a clinically validated diagnostic system.
- **Image Variability:** Differences in imaging equipment, image quality, and acquisition conditions may affect predictions.

### 🔬 Potential Improvements

- Explore fine-tuning of pretrained convolutional layers.
- Compare VGG16 with ResNet50, DenseNet121, and EfficientNet.
- Integrate Grad-CAM for visual explainability.
- Investigate additional image augmentation strategies.
- Evaluate model performance on independent datasets.

---

# 🔁 Reproducibility & Research Transparency

This repository provides a Jupyter Notebook containing the implementation and experimental workflow.

For reproducible experimentation, researchers should consider:

- Using the same dataset version.
- Maintaining consistent image preprocessing.
- Recording random seeds.
- Using identical training configurations.
- Documenting TensorFlow and Keras versions.
- Reporting complete classification metrics.

Reproducing identical results may require matching the original software environment and experimental settings.

---

# ⚠️ Medical Research Disclaimer

This project is developed for educational, academic, and research purposes.

The model is an experimental image classification system and is not intended to provide medical diagnoses, replace qualified healthcare professionals, or guide clinical treatment decisions.

Its predictions should not be used as a substitute for professional medical evaluation.

---

# 🚀 Future Improvements

Potential directions for further development include:

- [ ] 🧠 Fine-tuning pretrained VGG16 layers.
- [ ] 🔬 Comparing multiple CNN architectures.
- [ ] 🔥 Integrating Grad-CAM visualization.
- [ ] 📊 Adding ROC curves and AUC analysis.
- [ ] ⚖️ Investigating class imbalance handling.
- [ ] 🩺 Evaluating external chest X-ray datasets.
- [ ] 🌐 Developing an interactive prediction interface.
- [ ] 📈 Improving model interpretability and evaluation.

---

# 👨‍💻 About the Author

<div align="center">

## Shahid Azam

**Computer Science | Artificial Intelligence | Machine Learning**

MS Computer Science — Artificial Intelligence Specialization

Institute of Management Sciences, Peshawar, Pakistan

I am interested in developing intelligent systems and exploring practical applications of Artificial Intelligence, Deep Learning, Computer Vision, and Machine Learning.

### 🌐 Connect With Me

<a href="https://github.com/shahidazam2020-oss">
  <img src="https://img.shields.io/badge/GitHub-Follow_Me-181717?style=for-the-badge&logo=github" />
</a>

<a href="https://www.linkedin.com/in/shahid-azam-mughal-787b58235">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin" />
</a>

<a href="https://www.researchgate.net/profile/Shahid-Azam-5">
  <img src="https://img.shields.io/badge/ResearchGate-Research_Profile-00CCBB?style=for-the-badge&logo=researchgate" />
</a>

<a href="https://orcid.org/0009-0001-6825-8863">
  <img src="https://img.shields.io/badge/ORCID-Researcher_ID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" />
</a>

</div>

---

# 🤝 Contributing

Contributions, suggestions, and constructive feedback are welcome.

If you would like to improve this project:

1. Fork the repository.
2. Create a new branch.
3. Implement your improvements.
4. Commit your changes.
5. Submit a Pull Request.

### 💡 Contribution Areas

| Area | Possible Contributions |
|:---|:---|
| 🧠 Deep Learning | Alternative CNN architectures |
| 🔥 Explainable AI | Grad-CAM and visualization |
| 🖼️ Image Processing | Advanced preprocessing |
| 📊 Evaluation | Additional classification metrics |
| 📝 Documentation | Technical improvements |

<div align="center">

<a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16/issues">
  <img src="https://img.shields.io/badge/REPORT-Open_Issue-orange?style=for-the-badge&logo=github" />
</a>

<a href="https://github.com/shahidazam2020-oss/Pneumonia-Classification-Using-VGG16/pulls">
  <img src="https://img.shields.io/badge/CONTRIBUTE-Pull_Request-success?style=for-the-badge&logo=github" />
</a>

</div>

---

# 📜 License

No explicit license is currently specified for this repository.

Please contact the repository owner before redistributing or modifying project materials. Third-party datasets remain subject to their original licensing and usage terms.

---

# ⭐ Support This Project

If you find this project useful for your learning, research, or professional development:

- ⭐ Star this repository.
- 🍴 Fork the project.
- 🔗 Share it with your academic network.
- 💡 Suggest improvements.
- 🤝 Contribute to future development.

Your support and feedback are appreciated!

---

<div align="center">

### 🫁 Advancing Medical Imaging Through Deep Learning

**Artificial Intelligence • Computer Vision • Transfer Learning • Healthcare Research**

<br>

Made with ❤️ by **Shahid Azam**

<sub>© 2026 | Deep Learning & Medical Image Classification Project</sub>

</div>
