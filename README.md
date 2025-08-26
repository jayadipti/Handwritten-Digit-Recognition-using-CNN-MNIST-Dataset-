Handwritten Digit Recognition using CNN (MNIST Dataset)
Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset.
It demonstrates the end-to-end deep learning pipeline – from data preprocessing to model training, evaluation, visualization, and custom image prediction.

Achieved ~99% training accuracy and ~98% test accuracy.

 Key Features

Trains a CNN model on 60,000 training images & tests on 10,000 images

Visualizes predictions (True vs. Predicted labels)

Saves and tests with a custom digit image (test_image.png)

Includes confusion matrix & classification report for detailed evaluation

Beginner-friendly, easy to extend and deploy

Tech Stack

Language: Python 

Deep Learning: TensorFlow, Keras

Data Handling: NumPy, Pandas

Visualization: Matplotlib

Evaluation: scikit-learn


Dataset

The project uses the MNIST handwritten digit dataset available on Kaggle:

🔗 MNIST in CSV (Kaggle)-**https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download**

mnist_train.csv → 60,000 training images

mnist_test.csv → 10,000 testing images

Each row = label + 784 pixel values (28×28 image flattened)

Results

Training Accuracy: ~99%

Test Accuracy: ~98%


Evaluation Metrics

Along with accuracy, the script outputs:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Skills Demonstrated

Deep Learning (CNNs)

Data Preprocessing & Normalization

Model Training & Validation

Performance Evaluation (Confusion Matrix, Reports)

Custom Image Testing


Future Enhancements

Build a Flask/FastAPI web app where users can draw digits online

Extend to letters (A–Z) using EMNIST dataset

Deploy the model on Azure / Heroku / Docker for real-world use


