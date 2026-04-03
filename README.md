This project implements and compares Dense Autoencoders and Convolutional Autoencoders (CNNs) using the MNIST dataset. The goal is to study how different neural network architectures affect image reconstruction quality, as well as their computational efficiency during training and inference.


Before running the project, the required libraries should be installed using the command pip3 install -r requirements.txt.

Models
1. Dense Autoencoder 
- Fully connected (MLP-based) architecture
- Flattens the input image into a vector
- Learns compressed representations without spatial awareness
- Fast to train and computationally efficient

2. Convolutional Autoencoder (CNN)
- Uses convolutional and deconvolutional layers
- Preserves spatial structure of images
- Learns hierarchical features (edges, shapes, patterns)
- More computationally expensive but more accurate

Dataset:
- MNIST handwritten digits
- 28×28 grayscale images (resized to 256×256)
- Automatically downloaded via torchvision

Each model is trained using:
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Batch size: 128
- Epochs: configurable (default 5)
- Device: CPU

Results:

When running python3 main.py in the command line at the root of the repository, the original images are saved in the ./original_images folder, while reconstructed images from the Dense and CNN models are saved in ./Dense_images and ./CNN_images, respectively. These outputs allow for a visual comparison of reconstruction quality.

Based on the training results printed in the command line, the Dense Autoencoder achieves a final training loss of 0.0125 in approximately 6 seconds, while the CNN Autoencoder reaches a significantly lower loss of 0.0038 but requires about 62 seconds to train. It shows that CNN is more accurate because it preserves spatial information and learns meaningful image features, but it is slower due to the higher computational cost of convolution operations.This demonstrates a clear trade-off between speed and performance.

Evaluation on the test set shows a similar pattern. The Dense Autoencoder achieves a test MSE of 0.011324 with an inference time of 0.000149 seconds, whereas the CNN Autoencoder achieves a much lower test MSE of 0.002841 with an inference time of 0.000654 seconds. Although the CNN model is slightly slower during inference, both models remain very fast in practical use.

Overall, the results show that the CNN outperforms the Dense Autoencoder in terms of reconstruction accuracy, achieving approximately four times lower error. However, this improvement comes at the cost of substantially longer training time. The findings highlight the importance of spatial feature extraction in image-based tasks and demonstrate why CNN-based architectures are generally preferred for such problems.
