# Image-recognition

This is an image recognition program using python deep learning API Keras

# Run

It is recommended to run the train and test on the GPU.
Run the `train_4layer.py` or `train_6layer.py` for training, and then run the `test10.py` to test categorizing images into 10 classes. Batch size can be customized.

# Results

10 classes test

![test10](./images/test10.png)

Accuracy and loss

![accuracy4](./images/accuracy_4layer.png)
![loss4](./images/loss_4layer.png)

![accuracy6](./images/accuracy_6layer.png)
![loss6](./images/loss_6layer.png)

Confusion matrix

![cm4](./images/4_layer_cm.png)
![cm6](./images/4_layer_cm.png)

Visualizing cnn layer by layer

1st

![layer1](./images/1st_stitched_filters_6x6.png)

2nd

![layer2](./images/2nd_stitched_filters_6x6.png)

3rd

![layer3](./images/3rd_stitched_filters_6x6.png)


