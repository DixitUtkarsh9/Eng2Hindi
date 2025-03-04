# English to Hindi Language Translation

## Overview
This project focuses on building a sequence-to-sequence model for English to Hindi language translation using BART and MBART. Additionally, an encoder-decoder model with LSTM layers was implemented to compare performance. The models were trained on a dataset containing 127,000 rows and achieved high accuracy rates.

## Models Implemented
1. **BART & MBART (Transformer-based Model)**
   - Developed using pre-trained transformer models.
   - Achieved an accuracy of **97.67%**.

2. **LSTM-based Encoder-Decoder Model**
   - Implemented using Long Short-Term Memory (LSTM) layers.
   - Achieved an accuracy of **98.60%**.

## Dataset
- The dataset used contains **127,000** sentence pairs of English and Hindi text.
- Preprocessing included tokenization, padding, and vocabulary creation.

## Technologies Used
- Python
- PyTorch / TensorFlow
- Hugging Face Transformers
- Keras
- NLTK / SpaCy
- Jupyter Notebook / Google Colab

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/English-Hindi-Translation.git
cd English-Hindi-Translation

# Install dependencies
pip install -r requirements.txt
```

## Usage
To train and evaluate the models, run the following scripts:

```bash
# Train BART/MBART Model
python train_bart.py

# Train LSTM-based Model
python train_lstm.py

# Perform Translation
python translate.py --model bart --sentence "Hello, how are you?"
```

## Results
| Model | Accuracy |
|--------|------------|
| BART / MBART | 97.67% |
| LSTM Encoder-Decoder | 98.60% |

## Future Enhancements
- Fine-tune models with additional datasets for better generalization.
- Implement an interactive web interface for real-time translation.
- Explore Transformer-based architectures like T5 and GPT for improvement.

