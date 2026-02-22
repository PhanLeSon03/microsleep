# 🧠 Automatic Microsleep Detection using Wearable EEG

Deep learning framework for **automatic microsleep episode (MSE) detection** using EEG signals.

This repository implements **two dual-input neural network architectures**:

1. **STFT + Hand-Crafted Features**
2. **STFT + Raw Waveform (Learned Match Filters)**

**Best Result:**  
> **Cohen’s Kappa = 0.67**

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

Future extensions:
- 32-channel EEG
- fNIRS
- PPG
- Accelerometer
- Custom ADS1299 wearable system

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

Cohen’s Kappa ≈ **0.67**

---

[1] https://github.com/alexander-malafeev/microsleep-detection.git