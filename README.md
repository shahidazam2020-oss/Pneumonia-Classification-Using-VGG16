# 🫁 Pneumonia Classification Using VGG16

## 📌 Project Overview

**This project implements a deep learning based system for classifying chest X ray images into two categories:**

1. Normal
2. Pneumonia

*The project uses transfer learning with the VGG16 convolutional neural network. A pretrained VGG16 model is used as the base model, while additional custom layers are added for binary classification.*

**The main goal is to automatically identify whether a chest X ray image belongs to a healthy patient or a patient with pneumonia.**

## 📂 Dataset

*The project uses the Chest X Ray Pneumonia dataset.*

*The dataset contains chest X ray images organized into the following classes:*

• Normal

• Pneumonia

**The dataset is divided into training, validation, and testing directories.**

*The images are resized to:*

`224 × 224 × 3`

*This input size is compatible with the VGG16 model.*

## 🛠️ Technologies Used

**The project was developed using:**

• Python

• TensorFlow

• Keras

• VGG16

• NumPy

• Matplotlib

• Seaborn

• Scikit learn

• Kaggle

## 🔄 Project Workflow

**The complete workflow of the project is:**

**Dataset Download → Data Preprocessing → Data Augmentation → Image Visualization → VGG16 Transfer Learning → Model Training → Model Evaluation → Predictions → Confusion Matrix → Classification Report****

## 📊 Data Preprocessing

**All images are normalized by dividing pixel values by 255.**

```python
rescale = 1./255
```

**Data augmentation is applied to the training dataset to improve the model's ability to generalize.**

**The following augmentation techniques are used:**

• Rotation

• Zoom

• Horizontal flipping

**The validation and testing datasets are only rescaled.**

## 🖼️ Dataset Visualization

**Before training, sample chest X ray images are displayed from the training dataset.**

*Each image is labeled as either:*

• Normal

• Pneumonia

### Visualization

```text
┌────────────┬────────────┬────────────┐
│   Normal   │ Pneumonia  │   Normal   │
├────────────┼────────────┼────────────┤
│ Pneumonia  │   Normal   │ Pneumonia  │
├────────────┼────────────┼────────────┤
│   Normal   │ Pneumonia  │ Pneumonia  │
└────────────┴────────────┴────────────┘
```

**This visualization helps verify that the dataset is loaded correctly before model training.**

## 🧠 Model Architecture

**The VGG16 model is used as the pretrained feature extractor.**

*The ImageNet pretrained weights are loaded with:*

```python
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

**The pretrained VGG16 layers are frozen during training:**

```python
base_model.trainable = False
```

Custom layers are added on top of VGG16:

```text
Input Image
     ↓
VGG16 Pretrained Feature Extractor
     ↓
Flatten Layer
     ↓
Dense Layer, 512 neurons, ReLU
     ↓
Dropout, 0.5
     ↓
Dense Layer, 1 neuron, Sigmoid
     ↓
Prediction
```

## ⚙️ Model Configuration

**The model uses the following settings:**

| Parameter          | Value                |
| ------------------ | -------------------- |
| Base Model         | VGG16                |
| Pretrained Weights | ImageNet             |
| Input Size         | 224 × 224 × 3        |
| Hidden Layer       | 512 neurons          |
| Activation         | ReLU                 |
| Dropout            | 0.5                  |
| Output Activation  | Sigmoid              |
| Optimizer          | Adam                 |
| Learning Rate      | 0.0001               |
| Loss Function      | Binary Cross Entropy |
| Training Epochs    | 30                   |
| Batch Size         | 32                   |

## 📈 Training and Validation Results

**The model is trained for 30 epochs using the training dataset and validated on the validation dataset.**

*Two important graphs are generated during evaluation.*

### 1. Training and Validation Loss

**The loss graph shows how the model error changes during training.**

```text
Loss
 │\
 │ \
 │  \        Training Loss
 │   \____
 │
 │  \____   Validation Loss
 │
 └──────────────────────── Epochs
```

**A decreasing loss indicates that the model is learning useful patterns from the chest X ray images.**

### 2. Training and Validation Accuracy

**The accuracy graph compares the performance of the model on training and validation data.**

```text
Accuracy
 │                ╭──── Training Accuracy
 │           ╭────╯
 │      ╭────╯
 │ ╭────╯
 │ ╰────── Validation Accuracy
 │
 └──────────────────────── Epochs
```

**The actual graphs are generated automatically by the following code:**

```python
plot_learning_curves(history)
```

## 🧪 Model Evaluation

**After training, the model is evaluated using the independent test dataset.**

The evaluation calculates:

• Test Loss

• Test Accuracy

**The notebook prints the result using:**

```python
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.2f}')
```

### Result

```text
Test Accuracy: [Your Model Result]
```

**Replace `[Your Model Result]` with the exact accuracy produced when you run the notebook.**

## 🔍 Predictions

**The trained model predicts the probability that each chest X ray image belongs to the pneumonia class.**

*A threshold of 0.5 is used:*

```python
predicted_labels = (predictions > 0.5).astype(int)
```

Prediction values are interpreted as:

```text
0 → Normal
1 → Pneumonia
```

## 🔥 Confusion Matrix

A confusion matrix is generated to analyze the model's predictions.

```text
                    Predicted
                 Normal   Pneumonia

Actual Normal      TN         FP

Actual Pneumonia   FN         TP
```

Where:

• TN represents correctly predicted Normal images

• TP represents correctly predicted Pneumonia images

• FP represents Normal images incorrectly predicted as Pneumonia

• FN represents Pneumonia images incorrectly predicted as Normal

The project generates the confusion matrix using Seaborn:

```python
plot_confusion_matrix(cm, ['Normal', 'Pneumonia'])
```

### Confusion Matrix Visualization

Add your generated confusion matrix image to the repository:

```text
results/confusion_matrix.png
```

Then display it in the README:

```markdown
![Confusion Matrix](results/confusion_matrix.png)
```

## 📋 Classification Report

The model also generates a classification report containing:

• Precision

• Recall

• F1 Score

• Support

The report evaluates the performance for both classes:

```text
Normal
Pneumonia
```

The classification report is generated using:

```python
classification_report(
    true_labels,
    predicted_labels,
    target_names=['Normal', 'Pneumonia']
)
```

### Performance Metrics

| Metric    | Description                                             |
| --------- | ------------------------------------------------------- |
| Precision | Measures how many predicted cases are correct           |
| Recall    | Measures how many actual cases are correctly identified |
| F1 Score  | Balance between precision and recall                    |
| Support   | Number of samples in each class                         |

## 📊 Results

The final performance of the model should be presented here using your actual output.

| Metric              | Result          |
| ------------------- | --------------- |
| Test Accuracy       | Add your result |
| Test Loss           | Add your result |
| Normal Precision    | Add your result |
| Normal Recall       | Add your result |
| Pneumonia Precision | Add your result |
| Pneumonia Recall    | Add your result |
| Overall F1 Score    | Add your result |

## 📈 Project Visualizations

The repository should contain the following result images:

```text
results/
│
├── sample_images.png
├── training_validation_loss.png
├── training_validation_accuracy.png
└── confusion_matrix.png
```

These images can be displayed in the README.

### Sample Chest X Ray Images

```markdown
![Sample Images](results/sample_images.png)
```

### Training and Validation Curves

```markdown
![Learning Curves](results/learning_curves.png)
```

### Confusion Matrix

```markdown
![Confusion Matrix](results/confusion_matrix.png)
```

## 📁 Recommended Repository Structure

```text
Pneumonia-Classification-Using-VGG16/
│
├── Pneumonia Classification.ipynb
│
├── README.md
│
├── requirements.txt
│
├── results/
│   ├── sample_images.png
│   ├── learning_curves.png
│   └── confusion_matrix.png
│
└── LICENSE
```

## 🚀 How to Run the Project

1. Clone the repository.

2. Install the required libraries.

```python
pip install tensorflow numpy matplotlib seaborn scikit-learn kaggle
```

3. Download the Chest X Ray Pneumonia dataset.

4. Organize the dataset into training, validation, and test directories.

5. Run the Jupyter Notebook.

6. Train the VGG16 model.

7. Evaluate the model on the test dataset.

8. View the learning curves, confusion matrix, and classification report.

## 🎯 Conclusion

This project demonstrates the use of transfer learning for medical image classification. The pretrained VGG16 model extracts useful visual features from chest X ray images, while custom layers perform the final Normal versus Pneumonia classification.

The project includes data preprocessing, augmentation, image visualization, model training, performance evaluation, learning curves, confusion matrix analysis, and classification metrics. These results provide a complete overview of the model's performance on pneumonia classification.
