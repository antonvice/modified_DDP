# modified DDP
Deep Diffusion Process (DDP) Model
This is the code for the Deep Diffusion Process (DDP) model implemented in PyTorch. The DDP model is a generative model that generates data by performing a forward diffusion process followed by a reverse diffusion process.

# Requirements
PyTorch
Numpy
Matplotlib

# Usage
## Training
To train the model, you can use the train.py script. You can specify the training hyperparameters such as the number of epochs, learning rate, batch size, etc., as command-line arguments. For example:

''' python train.py --epochs 100 --lr 0.001 --batch_size 128 '''

## Evaluation
To evaluate the trained model, you can use the evaluate.py script. You need to specify the path to the saved model checkpoint as a command-line argument. For example:

''' python evaluate.py --checkpoint path/to/checkpoint.pth''' 

## Generating Samples
To generate samples from the trained model, you can use the generate.py script. You need to specify the path to the saved model checkpoint as a command-line argument. For example:

''' python generate.py --checkpoint path/to/checkpoint.pth '''

# Model Architecture
The DDP model consists of two networks: the forward diffusion step network and the reverse diffusion step network. The forward diffusion step network takes an input sample and generates a new sample by adding noise. The reverse diffusion step network takes the noisy sample and generates the original input sample. The model is trained by minimizing the difference between the generated and original samples.

In our implementation, we have added some additional features to the DDP model architecture to make it more robust, such as batch normalization, leaky ReLU activation, dropout, and weight initialization.

# Conclusion
The DDP model is a powerful generative model that can generate high-quality samples from complex distributions. With the added features in our implementation, the model becomes more robust and achieves better performance. This code provides an easy-to-use implementation of the DDP model in PyTorch, along with scripts for training, evaluation, and sample generation.
