# 🧠 Handwritten Digits Recognition using Neural Networks

This project is a simple yet powerful implementation of a **handwritten digit recognition system** using **TensorFlow** and the **MNIST dataset**. It trains a neural network to recognize digits (0–9) from images, and can also predict digits from your own custom handwritten images.

---

## 🚀 Features

- Trains a **deep neural network** with TensorFlow/Keras  
- Uses the **MNIST dataset** (70,000 images of handwritten digits)  
- Automatically saves and loads the trained model  
- Allows **custom handwritten digit prediction** using your own images  
- Displays predictions visually with **Matplotlib**

---

## 🧩 Project Structure

```
handwritten-digits-recognition/
│
├── digits/                     # Folder containing custom test images (digit1.png, digit2.png, etc.)
├── handwritten_digits.model     # Saved model (generated after training)
├── main.py                      # Main Python script (the code you provided)
└── README.md                    # Project documentation
```

---

## ⚙️ Requirements

Make sure you have the following Python libraries installed:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

---

## 🧠 How It Works

1. **Load the MNIST Dataset**  
   The dataset is automatically loaded via `tf.keras.datasets.mnist`.  
   It contains 60,000 training and 10,000 test grayscale images of digits.

2. **Normalize the Data**  
   Each image pixel value is normalized (scaled between 0 and 1).

3. **Build the Neural Network**
   - Input layer: Flattened 28x28 image (784 inputs)  
   - Hidden layers: Two dense layers with 128 neurons each, ReLU activation  
   - Output layer: 10 neurons with Softmax activation (for digits 0–9)

4. **Train the Model**  
   The model is trained for 3 epochs using the Adam optimizer and sparse categorical cross-entropy loss.

5. **Evaluate Performance**  
   The model’s accuracy and loss are displayed after evaluation on the test dataset.

6. **Save the Model**  
   The trained model is saved as `handwritten_digits.model` for later use.

7. **Predict Custom Digits**  
   Place your custom images (`digit1.png`, `digit2.png`, …) inside the `digits/` folder and the script will:
   - Read and preprocess them  
   - Predict the most likely digit  
   - Display the image and prediction using Matplotlib

---

## 🖼️ Custom Image Requirements

If you want to test your own digits:
- Save them as **PNG images** in the `digits/` folder  
- Name them like: `digit1.png`, `digit2.png`, `digit3.png`, etc.  
- The images should be:
  - **Black digits on a white background**
  - **28x28 pixels** (or close to it)
  - Grayscale (or will be converted automatically)

---

## ▶️ Usage

Run the script using:

```bash
python main.py
```

If `train_new_model = True`, the model will be trained from scratch.  
If set to `False`, it will load the saved model and go straight to predictions.

---

## 📊 Example Output

```
Epoch 1/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2602 - accuracy: 0.9257
Epoch 2/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1089 - accuracy: 0.9677
Epoch 3/3
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0743 - accuracy: 0.9770
313/313 [==============================] - 0s 1ms/step - loss: 0.0912 - accuracy: 0.9721
Validation Loss: 0.0912
Validation Accuracy: 0.9721

The number is probably a 4
```

---