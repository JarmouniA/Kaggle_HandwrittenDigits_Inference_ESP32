# Implementing a handwritten digits recognition application on an ESP32-CAM board using Tensorflow Lite for Microcontrollers.

The goal here is to perform inference on the 28000 test images, from this [Kaggle competition](https://www.kaggle.com/c/digit-recognizer), on an ESP32-CAM board and to submit the results in a csv file automatically to Kaggle using the Kaggle API.

The 28000 images will be served to an http Client on ESP32 by a Python http Server (over a wifi connection), and in the same way each inference result will be sent back to the server. When the processing of all the 28000 images is done, a POST request will be sent by the ESP32 to the server so that the inference results can be submitted to Kaggle.

-----------------------------
First of all, we need to build a machine learning model to perform the digits' recognition task. For that, I used TensorFlow and Keras in a [Kaggle notebook](https://www.kaggle.com/falconcode/digit-recognizer-tflite-micro) to build, train and evaluate a simple Convolutional Neural Network model using the MNIST dataset provided by the [Digit Recognizer kaggle competition ](https://www.kaggle.com/c/digit-recognizer).

After making sure that the model is working, we need to compress it so that it can be used for inference on memory-constrained devices like the ESP32. For that, we have to generate a TensorFlow Lite model, using one of the 2 techniques described in the [notebook](https://www.kaggle.com/falconcode/digit-recognizer-tflite-micro) :
- [Post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/training) : Generate a TFLite model from the baseline model with and without quantization.
- [Quantization-aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) : Generate the TFLite model from a quantization-aware model that was trained with the quantization step in mind.

Now that we have our TFLite model, we can generate a C file containing the model's weights and characteristics, which will be used by TensorFlow Lite for Microcontrollers, using the [Xxd tool](https://www.tutorialspoint.com/unix_commands/xxd.htm) as described in the last step in the notebook.

-----------------------------
Our TFLite micro is now ready to be deployed on [the edge](https://towardsdatascience.com/why-machine-learning-on-the-edge-92fac32105e6), we just need the TensorFlow Lite For Microcontrollers' library compiled for ESP32.
