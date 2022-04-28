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
Our TFLite Micro model is now ready to be deployed on [the edge](https://towardsdatascience.com/why-machine-learning-on-the-edge-92fac32105e6), we just need the [TensorFlow Lite For Microcontrollers](https://github.com/tensorflow/tflite-micro)' library compiled for ESP32. To get the library, the best option is to use the Docker image provided by the following [Github repository](https://github.com/atomic14/platformio-tensorflow-lite) (Thank you [@atomic14](https://github.com/atomic14)) to compile the library with the TensorFlow & [ESP-IDF](https://github.com/espressif/esp-idf) versions of your choice (I recommend using the latest stable versions).

The resulting TFLM library will be used with the Visual Studio Code extension [PlatformIO](https://platformio.org/) which really facilities the building process for ESP32 applications. Here is an official PlatformIO [guide](https://docs.platformio.org/en/latest/platforms/espressif32.html) that explains the different configuration options for ESP32 projects. Also, this project is build with the official [Espressif IoT Development Framework](https://docs.platformio.org/en/latest/frameworks/espidf.html#framework-espidf) (ESP-IDF) not the ESP32 Arduino Framework. The library must be placed in both the /components & /lib directories.

The [platformio.ini](platformio.ini) file already contains the necessary configuration options to build & deploy this project on the ESP32-CAM, and it will be detected automatically once the project in opened in VSCode with the PlatformIO extension installed.
