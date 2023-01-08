# Convolutional Vision Transformer (ConViT)

This is the pytorch code for ConViT(Improving Vision Transformers with Soft Convolutional Inductive Biases), which is one of the recent vision transformers that is more data efficient compared to the original Vision Transformer(ViT). 
I have trained this model using the CIFAR10 dataset to furhter validate the data efficiency of this model. 
Compared to the original Vision Transformer (ViT), the ConViT model needs fewer epochs to reach a decent performance. 

The final validation accuracy of ConViT trained on the CIFAR10 dataset with images of size 32*32 is 81% in 100 epochs with 2.3 million parameters.

Unlike the original ConViT model which has 12 layers (10 local and 2 global layers), this model is a more compressed version of ConViT with only 8 layers(6 local and 2 global layers).


![acc plot - Copy](https://user-images.githubusercontent.com/71688101/211201332-a1402d2b-f266-465e-9e5d-5a0605f22e58.png)

