# NBME_Daberta_v1 【inference】

This repository contains an inference pipeline for the NBME (National Board of Medical Examiners) Clinical Patient Notes scoring competition using DeBERTa model for medical text span extraction.

## 🎯 Overview

This project implements a **token-level binary classification** approach to identify relevant clinical information spans within patient notes. The model uses DeBERTa (Decoding-enhanced BERT with disentangled attention) to predict which text segments correspond to specific medical features.

### Key Features

- **Multi-fold Ensemble**: Combines predictions from 5 different model folds
- **Adaptive Threshold Selection**: Automatically finds optimal threshold for span extraction
- **Character-level Probability Mapping**: Maps token predictions to character-level probabilities
- **Robust Error Handling**: Comprehensive logging and error management
- **Memory Efficient**: Includes garbage collection and CUDA memory management

## 🏗️ Model Architecture

```
Input: [Feature Text] + [Patient Note History]
   ↓
DeBERTa Base Model
   ↓
Dropout (0.1)
   ↓
Linear Classifier (hidden_size → 1)
   ↓
BCEWithLogitsLoss
```

## 📁 Project Structure

```
.
├── nbme_deberta_v1_inference.py    # Main inference script
├── models/
│   ├── fold0.pt                    # Model checkpoints
│   ├── fold1.pt
│   └── ...
├── data/
│   ├── test.csv                    # Test data
│   ├── features.csv                # Feature definitions
│   └── patient_notes.csv           # Patient notes
└── submission.csv                  # Output predictions
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers pandas numpy tqdm scikit-learn
```

### Usage

1. **Prepare your data structure:**
   ```
   /kaggle/input/nbme-score-clinical-patient-notes/
   ├── test.csv
   ├── features.csv
   └── patient_notes.csv
   ```

2. **Place model checkpoints:**
   ```
   /kaggle/input/nbme-deberta-v1-train/nbme_ckpt/
   ├── fold0.pt
   ├── fold1.pt
   ├── fold2.pt
   ├── fold3.pt
   └── fold4.pt
   ```

3. **Run inference:**
   ```bash
   python nbme_deberta_v1_inference.py
   ```

## ⚙️ Configuration

Modify the `CFG` class to customize settings:

```python
class CFG:
    debug = False           # Debug mode
    num_workers = 0         # DataLoader workers
    model = "/kaggle/input/deberta/base"  # Model path
    batch_size = 8          # Batch size
    max_len = 512          # Max sequence length
    seed = 42              # Random seed
    n_fold = 5             # Number of folds
    trn_fold = [0,1,2,3,4] # Folds to use
```

## 🔧 Key Components

### 1. Model Class
- `DebertaForTokenBinary`: Custom DeBERTa wrapper for token-level binary classification

### 2. Data Processing
- `TestDataset`: Handles tokenization and data preparation
- `get_char_probs()`: Maps token predictions to character-level probabilities
- `get_predictions_from_char_probs()`: Converts probabilities to span predictions

### 3. Threshold Optimization
- `find_optimal_threshold()`: Automatically searches for optimal classification threshold
- `compute_f1_score()`: Evaluation metric calculation

### 4. Inference Pipeline
- Multi-fold model ensemble
- Batch processing with progress tracking
- Memory-efficient processing

## 📊 Output Format

The model outputs predictions in the following format:

```csv
id,location
0,"0 15;23 45"
1,"12 28"
2,""
```

Where `location` contains space-separated character indices indicating relevant text spans.

## 🎯 Performance Features

- **Automatic Threshold Selection**: Searches threshold range 0.45-0.56 to maximize performance
- **Ensemble Averaging**: Combines predictions from multiple folds for robustness
- **Efficient Memory Management**: Includes garbage collection and CUDA cache clearing
- **Comprehensive Logging**: Detailed progress tracking and error reporting

## 🔍 Model Details

- **Base Model**: microsoft/deberta-base
- **Max Sequence Length**: 512 tokens
- **Classification Head**: Single linear layer with dropout
- **Loss Function**: BCEWithLogitsLoss
- **Activation**: Sigmoid for probability output

## 📋 Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.0+
- pandas
- numpy
- tqdm
- scikit-learn

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NBME for providing the clinical patient notes dataset
- Microsoft for the DeBERTa model
- Hugging Face for the transformers library
