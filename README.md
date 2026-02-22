# Automatic Microsleep Detection

Deep learning framework for **automatic microsleep episode (MSE) detection** using EEG signals.

This repository implements **two dual-input neural network architectures**:

1. **STFT + Hand-Crafted Features**
2. **STFT + Raw Waveform (Learned Match Filters)**


### Download the dataset (https://zenodo.org/record/3251716)
```
bash download.sh
```

### Generate Tensorflow IO file
```
python TFRecord_gen_feature.py
```

###  Training
```
python 2In_TF_CNN_LSTM.py
```
or 
```
python TF_CNN_W.py
```

---

# 1. Microsleep Definition

Microsleep Episodes (MSEs) are defined as:

- Duration: **1–15 seconds**
- EEG slowing with **theta dominance (4–8 Hz)**
- Resembling NREM stage 1
- ≥ 80% eye closure confirmed by video

---

# 2. Dataset

Public dataset:

https://zenodo.org/record/3251716  

Channels used:
- O1-M1
- O2-M2
- EOG (LOC, ROC)
---

# 3. Model Architectures

---

## Model 1: STFT + Hand-Crafted Features

### Input 1: STFT Spectrogram
- Short-Time Fourier Transform
- Log power compression
- Channel-wise normalization
- Hop size: 32 (160 ms)
- FFT window: 128 (640 ms)

Shape: (128, 65, 4): sequence, SFTF, channel 

---

### Input 2: Hand-Crafted Features

Computed per window:640 ms

**EEG Features (21 features)**
- Band power (δ, θ, α, β)
- Band power ratio
- Spectral entropy

**EOG Features (8 features)**
- Slow eye movement energy
- Blink amplitude
- Low-frequency power

Shape: (32, 2*(21+8)): sequence, feature

### Output: 
- (32, 1): 32 sequence of 640ms

---

### Performance
Cohen’s Kappa ≈ **0.67** (published model CNN_LSTM: 0.65 [1])
```
              precision    recall  f1-score   support

         0.0       0.97      0.97      0.97     38082
         1.0       0.70      0.71      0.70      3582

    accuracy                           0.95     41664
   macro avg       0.83      0.84      0.84     41664
weighted avg       0.95      0.95      0.95     41664

computes Cohen kappa per class: [0.67459876 0.67459876]
```
---

## Model 2: STFT + Raw Waveform (Learned Match Filters)

Main contribution of this work.

Instead of manual feature engineering, the model **learns waveform templates automatically**.

---

### Input 1: STFT (same as Model 1)

---

### Input 2: Raw Waveform

Shape: (32, 400, 4) : sequence, sample, channel


### Output: 
- (32, 1): 32 sequence of 640ms

---

### Performance

Cohen’s Kappa ≈ **0.66**
```
              precision    recall  f1-score   support

         0.0       0.99      0.94      0.96     76132
         1.0       0.59      0.86      0.70      7164

    accuracy                           0.94     83296
   macro avg       0.79      0.90      0.83     83296
weighted avg       0.95      0.94      0.94     83296

computes Cohen kappa per class
[0.66121442 0.66121442]
```
---

[1] https://github.com/alexander-malafeev/microsleep-detection.git