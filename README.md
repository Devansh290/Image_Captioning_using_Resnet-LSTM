# 🖼️ Image Captioning using CNN + LSTM with Attention

A deep learning project that automatically generates descriptive captions for images by integrating **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks enhanced with an **attention mechanism**. Built using **PyTorch** and trained on the **Flickr8k** dataset.


## 📌 Table of Contents
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Decoding Strategies](#decoding-strategies)
- [Evaluation](#evaluation)
- [Results](#results)
- [Challenges](#challenges)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

---

## 🧠 Abstract

This project implements an image captioning system using an **Encoder-Decoder architecture**. It uses:
- **ResNet-50** as the encoder for feature extraction
- **LSTM** with **attention** as the decoder to generate text
- **Beam search** and **greedy decoding** for inference
- Evaluation via **BLEU scores**

---

## 🗂️ Dataset

We used the **Flickr8k dataset**, consisting of:
- 8,000 images in JPEG format
- Each image annotated with 5 human-written captions

### Preprocessing:
- Images resized and normalized for ResNet-50
- Captions lowercased, tokenized, and padded
- Vocabulary built from training captions (removing rare words)

---

## 🧩 Model Architecture

### Encoder
- Pretrained **ResNet-50** without the final classification layer
- Added a custom `nn.Linear` layer for 256-dim embeddings

### Attention
- Computes weights over spatial image regions
- Generates context vectors using weighted sum

### Decoder
- Embedding layer → LSTM (hidden size 512) → Fully connected layer → Vocabulary logits
- Each step combines word embedding + attention context

---

## ⚙️ Training Details

- Framework: **PyTorch**
- Platform: **Google Colab (with GPU)**
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Batch Size: 64
- Training:
  - 10 epochs with frozen encoder
  - 4 additional epochs after fine-tuning encoder
- Gradient Clipping: Used to prevent exploding gradients

---

## 🧾 Decoding Strategies

### 1. Greedy Decoding
- Picks the most probable word at each step
- Fast but can miss optimal sequences

### 2. Beam Search
- Beam size = 3
- Maintains top k candidates per step
- Produces more fluent and accurate captions

---

## 📊 Evaluation

### Qualitative:
- Visual inspection of generated captions vs images

### Quantitative:
- BLEU scores used for comparing predicted captions with ground truth

---

## 🏁 Results

- Beam Search outperformed Greedy decoding
- Generated captions were often:
  - Contextually accurate
  - Grammatically sound

### Example Output:
- **Greedy**: "a dog is running in the grass"
- **Beam Search**: "a brown dog running through the grass"

---

## ⚠️ Challenges

- Limited computational resources (Google Colab timeouts)
- Small dataset size (Flickr8k)
- Occasional overfitting and attention misfocus
- Mislabeling and repetitive output in some cases

---

## ✅ Conclusion

This project demonstrates the effectiveness of combining CNNs, LSTMs, and attention for generating meaningful image captions. The approach bridges computer vision and natural language processing in a practical application.

---

## 🔭 Future Work

- Training on larger datasets (e.g., MS COCO)
- Using transformer-based models (e.g., ViT + GPT)
- Enhancing attention mechanisms
- Real-time image captioning with web camera input

---

## 📁 Folder Structure

```bash
📦Image-Captioning
 ┣ 📂images
 ┣ 📂models
 ┣ 📜caption_utils.py
 ┣ 📜model.py
 ┣ 📜train.py
 ┣ 📜predict.py
 ┣ 📜README.md
