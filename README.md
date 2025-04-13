# CNN for Handwritten Digit Classification  

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) using the MNIST dataset. The model is built using TensorFlow and Keras, achieving high accuracy in recognizing digits.  

## Project Overview  
- Dataset: MNIST (handwritten digits)  
- Model Architecture: Convolutional Neural Network (CNN)  
- Tools Used: TensorFlow, Keras, NumPy, Matplotlib  
- Training Epochs: 5  
- Evaluation Metric: Accuracy  

## Dataset  
The MNIST dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels in grayscale. The images represent handwritten digits from 0 to 9.  

## Installation & Setup  
To run this project, install the required dependencies:  
pip install tensorflow numpy matplotlib  
 
 

Then, run the Python script:  
python cnn_mnist.py  
 
 

## Model Architecture  
The CNN model consists of the following layers:  
1. Conv2D (32 filters, 3x3 kernel, ReLU activation)  
2. MaxPooling2D (2x2 pool size)  
3. Conv2D (64 filters, 3x3 kernel, ReLU activation)  
4. MaxPooling2D (2x2 pool size)  
5. Conv2D (64 filters, 3x3 kernel, ReLU activation)  
6. Flatten (convert feature maps into a single vector)  
7. Dense (64 neurons, ReLU activation)  
8. Dense (10 neurons, Softmax activation for classification)  

## Results  
The trained model achieves high accuracy in classifying digits.  
Example output:  
Test accuracy: 0.98  
Predicted Label: 7  
 
 

## Visualization  
The model can predict individual test images. Below is an example of a digit 7 correctly classified:  

![Sample Output](https://github.com/Manar-99/CNN_Project/blob/main/sample.png)  

## Reference Paper  
This implementation is based on the research paper:  
"An Introduction to Convolutional Neural Networks" â [Read on arXiv](https://arxiv.org/abs/1511.08458)  

## GitHub Repository  
[Project Repository](https://github.com/Manar-99/CNN_Project.git)
