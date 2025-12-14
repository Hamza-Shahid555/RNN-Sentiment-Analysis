# Sentiment Analysis: Integer Encoding vs. Word Embeddings

REMEMBER THIS 
HERE THE FOCUS JUST HOW RNN WORK 
NOT ON ACCUERACY OF A MODEL 

📌 Project Overview
This repository implements two different approaches to Sentiment Analysis using the **IMDB Movie Reviews dataset**. 
The goal is to classify reviews as positive or negative using Recurrent Neural Networks (SimpleRNN).

The project serves as a technical demonstration of why **Word Embeddings** are essential for NLP tasks compared to basic **Integer Encoding**.

📂 Files Description

### 1. `integer_encoding_simplernn.ipynb`
* **Approach:** Uses raw integer sequences directly as input for the RNN.
* **Architecture:** `Input (Integer Sequence)` $\rightarrow$ `SimpleRNN` $\rightarrow$ `Dense (Sigmoid)`.
* **Outcome:** The model fails to learn meaningful patterns because it interprets word indices as having ordinal magnitude (e.g., word #200 is "greater" than word #10).
* **Performance:** The validation accuracy stagnates around **50%** (random guessing).

### 2. `sentiment_analysis_simplernn.ipynb`
* **Approach:** Maps integer indices to dense vectors using a Keras Embedding layer before passing them to the RNN.
* **Architecture:** `Embedding Layer` $\rightarrow$ `SimpleRNN` $\rightarrow$ `Dense (Sigmoid)`.
* **Outcome:** The model successfully captures semantic relationships between words.
* **Performance:** Achieves a training accuracy of roughly **88%** and validation accuracy of approx **79-80%** within 5 epochs.

## 🛠️ Tech Stack
* **Python**
* **TensorFlow / Keras** (Sequential API, Layers: Embedding, SimpleRNN, Dense)
* **NumPy**

## 📊 Performance Comparison

| Metric | Integer Encoding Model | Word Embedding Model |
| :--- | :--- | :--- |
| **Input Shape** | `(50, 1)` | `(50, output_dim)` (via Embedding) |
| **Train Accuracy** | ~50.8% | ~88.2% |
| **Val Accuracy** | ~50.6% | ~79.0% |
| **Loss Function** | Binary Crossentropy | Binary Crossentropy |
| **Optimizer** | Adam | Adam |

## 🧠 Key Takeaways
1.  **Integer Encoding Limitations:** Feeding raw integers into a neural network for text creates false numerical relationships between words, preventing the RNN from learning.
2.  **Power of Embeddings:** The Embedding layer transforms discrete integers into continuous vector spaces where semantic meaning is learned, leading to a massive jump in accuracy (from ~50% to ~80%).

## 🚀 How to Run
1. Clone the repository.
2. Ensure you have TensorFlow installed:
   ```bash
   pip install tensorflow numpy
