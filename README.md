# 🧠 EEG-Based Mental Workload Classification

### Machine Learning for EEG Signal Analysis and Cognitive Workload Assessment

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-orange)](#)
[![EEG](https://img.shields.io/badge/EEG-Signal%20Analysis-purple)](#)
[![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![MNE](https://img.shields.io/badge/MNE-EEG%20Analysis-blue)](https://mne.tools/)

---

## 📌 Overview

This project presents a **machine-learning approach for classifying mental workload from EEG (Electroencephalography) signals**.

The system processes EEG recordings, extracts frequency-domain features from multiple EEG channels, selects the most informative features, and trains a Support Vector Machine (SVM) classifier to distinguish between different levels of mental workload.

The project also integrates the trained machine-learning model into a **graphical user interface (GUI)** that allows users to process EEG data and obtain workload predictions.

The predicted workload levels are represented as:

- 🟢 **Low**
- 🟡 **Medium**
- 🔴 **High**

The repository therefore combines:

**EEG Signal Processing + Feature Engineering + Machine Learning + Model Evaluation + GUI Application**

---

# 🎯 Project Objective

The main objective of this project is:

> **To investigate whether EEG signal characteristics can be used to automatically classify different levels of mental workload using machine learning.**

The project follows a complete machine-learning pipeline from raw EEG data to an application-level prediction.

---

# 🧠 What is Mental Workload?

Mental workload describes the amount of cognitive effort required to perform a task.

Understanding mental workload is important in areas such as:

- Human–Computer Interaction
- Brain–Computer Interfaces
- Neuroergonomics
- Cognitive-state monitoring
- Adaptive interfaces
- Human–AI interaction
- Safety-critical systems

EEG provides a non-invasive way of recording electrical brain activity. This project investigates whether patterns within EEG signals can provide useful information for distinguishing different workload levels.

---

# 🔬 Methodology

The project follows the following pipeline:

```text
                  EEG Dataset
                       │
                       ▼
              EEG Data Loading
                       │
                       ▼
              Data Standardisation
                       │
                       ▼
           Frequency-Domain Analysis
                       │
                       ▼
              Band-Power Features
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Delta          Theta          Alpha
        │              │              │
        └──────────────┼──────────────┘
                       │
                 Beta + Gamma
                       │
                       ▼
               Feature Selection
                 SelectKBest
                       │
                       ▼
               SVM Classification
                       │
                       ▼
          Leave-One-Group-Out Testing
                       │
                       ▼
            Mental Workload Prediction
                       │
                       ▼
                 GUI Application
```

# 📊 EEG Data Processing

The training pipeline processes EEG recordings stored in MATLAB `.mat` files.

The implementation loads the cleaned EEG data and associated labels from the dataset structure.

The preprocessing pipeline includes:

1. Loading EEG recordings
2. Extracting cleaned EEG data
3. Extracting workload labels
4. Standardising the EEG signals
5. Encoding categorical labels
6. Extracting frequency-domain features

The current implementation uses a **128 Hz sampling frequency**.

---

# 🧠 EEG Frequency Bands

The project extracts EEG power information from five frequency bands:

| Frequency Band | Frequency Range |
|---|---|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

These features are calculated using **Welch's power spectral density estimation** through **MNE-Python**.

The frequency-domain features are calculated for the EEG channels and then transformed into a machine-learning feature representation.

---

# 📡 EEG Channels

The implementation works with **14 EEG channels**.

For each channel, the system extracts power features from:

- Delta
- Theta
- Alpha
- Beta
- Gamma

This produces a multi-dimensional representation of the EEG signal that can be used by the machine-learning model.

# 🧪 Feature Engineering

Feature engineering is a major component of the project.

The pipeline is:

```text
EEG Signal
     │
     ▼
Standardisation
     │
     ▼
Power Spectral Density
     │
     ▼
Frequency Band Power
     │
     ▼
Feature Matrix
     │
     ▼
Feature Selection
```
The project uses **SelectKBest** with an **ANOVA F-statistic (`f_classif`)** to select the most informative features.

The current implementation selects the **top 100 features** from the available feature set.

# 🤖 Machine Learning Model

The main classifier implemented in the training pipeline is a:

## Support Vector Machine (SVM)

The model uses:

```text
StandardScaler
      +
SVM
      │
      └── Polynomial Kernel
```
The classifier is implemented using sklearn.svm.SVC with:
```python
SVC(
    kernel='poly',
    decision_function_shape='ovr'
)
```
The model is combined with StandardScaler inside a scikit-learn Pipeline.

# 🔄 Model Evaluation

Because EEG data can vary considerably between individuals, evaluating performance across subjects is particularly important.

The project uses:

## Leave-One-Group-Out Cross-Validation

Subjects are treated as groups during evaluation.

In each iteration:

```text
Multiple Subjects
       │
       ▼
Training Data
       │
       ▼
SVM
       │
       ▼
Held-Out Subject
       │
       ▼
Prediction
       │
       ▼
Accuracy
```
The implementation calculates an accuracy score for each held-out subject and then calculates the average accuracy across the evaluation process.

This evaluation strategy helps test whether the model can generalise across different subjects rather than only learning patterns from the same individuals used during training.

# 💾 Trained Model

After training, the SVM pipeline is saved using Python's **pickle** functionality.

The trained model is saved as:

```text
models/svm_loso.pkl
```
This allows the trained classifier to be integrated into the application rather than retraining the model every time the application is used.

🖥️ Application

The project includes a graphical user interface designed to make the machine-learning system accessible through an application rather than requiring users to interact directly with the training code.

The GUI is implemented using Tkinter / CustomTkinter and integrates the EEG processing and prediction functionality.

The application includes functionality for:

User login
EEG data processing
Mental workload prediction
Prediction visualisation
Prediction history
Session management
Local database interaction
📈 Workload Prediction

The application converts the model's numerical prediction into an interpretable workload level:
```text
Prediction
    │
    ├── 0 → LOW
    │
    ├── 1 → MEDIUM
    │
    └── 2 → HIGH
```
The application also includes a visual gauge representing the predicted workload level.

This makes the machine-learning output easier for a user to interpret.

# 🗂️ Prediction History

The application includes functionality for storing and retrieving prediction history.

The project uses **SQLite** for local data storage.

The database structure includes information related to:

- Users
- Sessions
- Subjects
- Predictions
- Prediction timestamps

The history functionality allows predictions to be associated with the logged-in user and retrieved later.
# 🏗️ System Architecture

The overall system can be represented as:

```text
                    ┌──────────────────┐
                    │    EEG Dataset   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Processing  │
                    │ & Standardisation│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature          │
                    │ Extraction       │
                    │                  │
                    │ Delta            │
                    │ Theta            │
                    │ Alpha            │
                    │ Beta             │
                    │ Gamma            │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature          │
                    │ Selection        │
                    │ SelectKBest      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ SVM Classifier   │
                    │ Polynomial Kernel│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Workload         │
                    │ Prediction       │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │        GUI Application       │
              │                              │
              │   Low / Medium / High        │
              │                              │
              │   Prediction History         │
              └──────────────────────────────┘
```

# 📁 Repository Structure

```text
EEG-Based-Mental-Workload-Classification/
│
├── assets/
│   └── Application assets and interface resources
│
├── models/
│   └── Trained machine-learning models
│
├── db.py
│   └── SQLite database functionality
│
├── gui.py
│   └── Main GUI functionality
│
├── history.py
│   └── Prediction history functionality
│
├── home.py
│   └── Application home interface
│
├── main.py
│   └── Application entry point
│
├── model.py
│   └── EEG preprocessing and prediction functionality
│
├── nav.py
│   └── Application navigation
│
├── training.py
│   └── EEG processing, feature extraction and model training
│
├── icon.ico
│   └── Application icon
│
├── .gitignore
├── .gitattributes
└── README.md
```

# 🛠️ Technologies & Libraries

## Programming

- Python

## EEG & Signal Processing

- MNE-Python
- NumPy
- SciPy
- Welch Power Spectral Density

## Machine Learning

- Scikit-learn
- Support Vector Machines
- StandardScaler
- SelectKBest
- ANOVA F-statistics
- Leave-One-Group-Out cross-validation

## Data Processing

- Pandas
- NumPy
- MATLAB `.mat` file processing

## Application Development

- Tkinter
- CustomTkinter
- Pillow
- SQLite
- Matplotlib

## Model Persistence

- Pickle
- Joblib

  
