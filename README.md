# Digit Recognition System Using PCA and Logistic Regression

## Overview
This project implements a digit recognition system using Principal Component Analysis (PCA) for dimensionality reduction and Logistic Regression for classification. The goal is to accurately recognize handwritten digits from images.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
The digit recognition system leverages PCA to reduce the dimensionality of the input data, which consists of images of handwritten digits. Logistic Regression is then employed to classify the reduced feature set into the corresponding digit labels.

## Getting Started
To get started with this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Hashir-Ali-Shah/Digit-Recognition-Clustering-PCA-Logistic-Regression.git
   ```

2. Navigate into the project directory:
   ```bash
   cd Digit-Recognition-Clustering-PCA-Logistic-Regression
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To train and test the digit recognition model, run the following command:
```bash
python main.py
```
This will execute the main program which handles data loading, model training, and testing.

## Results
The model's performance will be evaluated on the test dataset, and metrics such as accuracy will be displayed.

## Conclusion
This digit recognition system effectively demonstrates the application of PCA for feature reduction and Logistic Regression for classification. Further improvements can be made by experimenting with different algorithms and hyperparameters.

## License
This project is licensed under the MIT License. See the LICENSE file for details.