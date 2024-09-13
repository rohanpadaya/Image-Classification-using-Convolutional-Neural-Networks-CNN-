# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

## Overview
This project demonstrates the application of Convolutional Neural Networks (CNN) to classify images from the CIFAR-10 dataset. The goal is to develop various CNN models, including baseline models and more advanced architectures like ResNet, to achieve high accuracy in image classification.

## Dataset
- **Source**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.
- **Classes**:
  1. Airplane
  2. Automobile
  3. Bird
  4. Cat
  5. Deer
  6. Dog
  7. Frog
  8. Horse
  9. Ship
  10. Truck

- **Download the dataset**: [CIFAR-10 Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv)

## Project Files
- **CNN_code.ipynb**: Jupyter Notebook containing the code for training and evaluating the CNN models.
- **CNN_CIFAR10.pdf**: Detailed project report that discusses the methodology, models, and results.
- **Dataset**: (To be uploaded) The CIFAR-10 dataset used for training and testing.

## Models Developed
1. **Baseline CNN Model**:
   - Architecture: 8 layers, 3 convolutional layers followed by max-pooling layers, and 2 fully connected layers.
   - Validation Accuracy: 70.04%

2. **Data Augmented CNN Model**:
   - Techniques: Rotation, zooming, and horizontal flipping.
   - Validation Accuracy: 73.45%

3. **Grid Search CNN Model**:
   - Hyperparameters Tuned: Optimizer (SGD, Adam), Batch Size (32, 64), Epochs (10, 20).
   - Best Validation Accuracy: 72.18%

4. **ResNet-50 Model**:
   - Architecture: 50-layer deep network with residual connections to solve the vanishing gradient problem.
   - Validation Accuracy: 75.08%

5. **ResNet-50 with L2 Regularization**:
   - Regularization: L2 Regularization to avoid overfitting.
   - Best Validation Accuracy: 80.66%

## Methodology
1. **Data Preprocessing**: Images were normalized to [0,1] range. The data was split into training, validation, and testing sets.
2. **Model Development**: Various CNN architectures were developed and trained on the CIFAR-10 dataset.
3. **Hyperparameter Tuning**: Grid search was applied to optimize the model parameters.
4. **Model Evaluation**: Each model was evaluated based on its accuracy on the validation set, and the best model was tested on the test set.

## Results
- **Best Model**: ResNet-50 with L2 Regularization.
- **Test Accuracy**: 79.73%

## How to Run the Project
1. Clone the repository: `git clone https://github.com/yourusername/cifar10-image-classification.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook: Open `CNN_code.ipynb` in Jupyter and execute the cells to train and evaluate the models.

## Acknowledgments
This project was developed as part of the coursework at The University of Adelaide. The dataset was provided by [Kaggle](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv).
