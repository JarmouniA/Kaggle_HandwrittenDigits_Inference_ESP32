# Implementing a handwritten digits recognition application on an ESP32-CAM board using Tensorflow Lite for Microcontrollers.

The goal is to perform inference on the 28000 test images, from the [Kaggle competition](https://www.kaggle.com/c/digit-recognizer), on the ESP32-CAM and submit the results in a csv file automatically to Kaggle using the Kaggle API.

The 28000 images will be served to an http Client on ESP32 by a Python http Server, and in the same way each inference result will be sent to the server. When the processing of all the 28000 images is done, a POST request will be sent by the ESP32 to the server so that the inference results will be submitted to Kaggle.
