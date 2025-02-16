# Hyperspectral Image Classification

This repository contains code for hyperspectral image (HSI) classification using a Convolutional Neural Network (CNN) approach. The implementation focuses on the Indian Pines dataset and includes data preprocessing, augmentation, model training capabilities, and comprehensive visualization tools.

## Features

- Data loading from Kaggle datasets
- Patch-based data preprocessing
- Class balancing through oversampling
- Data augmentation with flipping and rotation
- CNN model implementation for classification
- Performance evaluation and reporting
- Comprehensive visualization capabilities for HSI data and results

## Prerequisites

```python
numpy
pathlib
kagglehub
tensorflow
sklearn
scipy
matplotlib
seaborn
```

## Dataset

The implementation supports multiple hyperspectral datasets:
- Indian Pines dataset
- Pavia University dataset

Datasets are automatically downloaded using kagglehub:
```python
pavia_path = kagglehub.dataset_download('mlxlx0000/paviadata')
indian_pines = kagglehub.dataset_download('freeman2147/1992-indian-pines')
```

## Data Preprocessing

### Patch Creation
The code creates patches from the hyperspectral image using a sliding window approach:
- Includes zero padding for border pixels
- Configurable window size (default: 5x5)
- Option to remove patches with zero labels

### Class Balancing
Implements oversampling of minority classes to handle class imbalance:
- Calculates class distributions
- Repeats samples from minority classes
- Includes random permutation for balanced training

### Data Augmentation
Supports multiple augmentation techniques:
- Vertical flipping
- Horizontal flipping
- Random rotation (30° intervals between -180° and 180°)

## Model Architecture

The CNN model consists of:
- Two convolutional layers with ReLU activation
- Dropout layers for regularization
- Dense layers for classification
- Softmax output layer for 16 classes

```python
model = Sequential([
    Conv2D(C1, (3, 3), activation='relu'),
    Conv2D(3*C1, (3, 3), activation='relu'),
    Dropout(0.25),
    Flatten(),
    Dense(6*numComponents, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='softmax')
])
```

## Visualization Tools

### Dataset Visualization
- False color composition of hyperspectral bands
- Ground truth visualization with custom colormaps
- Class distribution plots
- Spectral signature plots for different classes

### Training Visualization
- Learning curves (loss and accuracy)
- Class-wise accuracy plots
- Training progress monitoring
- Batch visualization during training

### Results Visualization
- Confusion matrix heatmap
- Classification map overlay
- Prediction probability maps
- Error analysis visualization
- ROC curves and precision-recall curves
- t-SNE visualization of feature spaces

### Patch Visualization
- Original patch display
- Augmented patch comparison
- Feature map visualization
- Class activation mapping (CAM)

## Training

The model is trained with:
- Adam optimizer
- Categorical crossentropy loss
- Batch size of 32
- Configurable number of epochs

## Evaluation

Includes comprehensive evaluation metrics:
- Classification report with per-class performance
- Confusion matrix
- Test loss and accuracy
- Support for 16 different crop classes

## Usage

1. Load and preprocess the data:
```python
XPatches, yPatches = createPatches(indian_pines_pca, indian_pines_gt, windowSize=5)
X_train, X_test, y_train, y_test = train_test_split(XPatches, yPatches, test_size=0.2)
```

2. Apply oversampling and augmentation:
```python
X_train, y_train = oversampleWeakClasses(X_train, y_train)
X_train = AugmentData(X_train)
```

3. Train the model:
```python
model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS)
```

4. Evaluate performance:
```python
classification, confusion, test_Loss, test_accuracy = reports(X_test, y_test)
```

5. Visualize results:
```python
# Plot confusion matrix
plot_confusion_matrix(confusion, target_names, title='Confusion Matrix')

# Display classification map
plot_classification_map(prediction_map, ground_truth)

# Show spectral signatures
plot_spectral_signatures(X_train, y_train, class_names)

# Visualize feature maps
plot_feature_maps(model, X_test[0])
```

## Classes

The model classifies 16 different types of land cover:
- Alfalfa
- Corn-notill
- Corn-mintill
- Corn
- Grass-pasture
- Grass-trees
- Grass-pasture-mowed
- Hay-windrowed
- Oats
- Soybean-notill
- Soybean-mintill
- Soybean-clean
- Wheat
- Woods
- Buildings-Grass-Trees-Drives
- Stone-Steel-Towers

## Visualization Best Practices

1. **Color Schemes**:
   - Use colorblind-friendly palettes
   - Maintain consistent color mapping across visualizations
   - Use appropriate colormaps for different data types

2. **Layout**:
   - Include proper titles and labels
   - Add colorbar for reference
   - Use appropriate figure sizes for clarity

3. **Interactive Features**:
   - Enable zoom functionality for detailed inspection
   - Provide hoverable tooltips for data points
   - Allow toggling between different visualization modes

4. **Export Options**:
   - Support multiple format exports (PNG, PDF, SVG)
   - Maintain high resolution for publications
   - Include scale bars where appropriate