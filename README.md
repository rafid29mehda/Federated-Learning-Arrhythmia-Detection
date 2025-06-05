# Federated Learning for Arrhythmia Detection

This project implements a **Federated Learning (FL)** framework for detecting cardiac arrhythmias using ECG data from the MIT-BIH Arrhythmia Database. By employing FL, the model trains on decentralized datasets across multiple clients (e.g., hospitals) without sharing sensitive patient data, ensuring privacy while achieving high classification performance. The project uses the **Flower (flwr)** framework for FL, **TensorFlow** for model development, and **wfdb** for ECG data processing.

## Project Overview

The objective is to classify ECG beats as normal or abnormal using a 1D Convolutional Neural Network (CNN) trained via federated learning. The MIT-BIH Arrhythmia Database provides ECG signals and annotations for training and evaluation. The project simulates five clients, each holding a subset of the dataset, and aggregates model updates using a custom Federated Averaging (FedAvg) strategy.

### Key Features
- **Data Preprocessing**: Extracts 1-second ECG segments (720 samples at 360 Hz) around annotated beats, labeled as normal ('N') or abnormal (e.g., 'L', 'R', 'V').
- **Federated Learning**: Simulates five clients with local datasets, aggregating model weights on a central server.
- **CNN Model**: A 1D CNN with two convolutional layers, max-pooling, and dense layers for binary classification.
- **Privacy-Preserving**: No raw data is shared, only model weights are exchanged.
- **Evaluation**: Reports test loss, accuracy, precision, recall, and F1-score on a held-out test set.

## Methodology

### Dataset
The **MIT-BIH Arrhythmia Database** contains 48 half-hour ECG recordings with annotated beats, sampled at 360 Hz. The dataset is downloaded using `wfdb` to `/content/mitdb`. Beats are classified as:
- **Normal**: 'N'
- **Abnormal**: 'L', 'R', 'V', '/', 'A', 'f', 'F', 'j', 'a', 'E', 'J', 'e', 'S'

**Preprocessing**:
- ECG signals are segmented into 1-second windows (720 samples) centered on annotated beats.
- Segments are labeled as normal (0) or abnormal (1).
- Data is split across five clients and a separate test set (records 230–234).

### Data Splitting
- **Clients**: 44 records are divided among 5 clients, resulting in:
  - Client 1: 24,595 samples
  - Client 2: 22,976 samples
  - Client 3: 22,357 samples
  - Client 4: 19,705 samples
  - Client 5: 19,708 samples
- **Test Set**: 11,428 samples from records 230–234.

### CNN Model Architecture
The 1D CNN model is defined as follows:
```plaintext
Input: (720, 1) ECG segment
Conv1D: 32 filters, kernel size 3, ReLU
MaxPooling1D: Pool size 2
Conv1D: 64 filters, kernel size 3, ReLU
MaxPooling1D: Pool size 2
Flatten
Dense: 64 units, ReLU
Dense: 1 unit, Sigmoid
Loss: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
```
- **Parameters**: 735,553
- **Output Shape**: Detailed in model summary (see code output).

### Federated Learning Setup
- **Framework**: Flower (flwr) with a custom FedAvg strategy.
- **Clients**: 5 simulated clients, each training a local CNN model for 5 epochs per round.
- **Server**: Aggregates weights over 10 rounds, saving final weights.
- **Simulation**: Uses Ray for distributed computation with 1 CPU per client.

## Prerequisites

- Python 3.11+
- Install dependencies:
  ```bash
  pip install flwr tensorflow wfdb
  pip install -U "flwr[simulation]"
  pip install ray[default]
  ```

## Project Structure

The Python script (`federated_learning_arrhythmia_detection.py`) includes:
1. **Library Installation**: Installs required packages.
2. **Dataset Download**: Fetches MIT-BIH Arrhythmia Database.
3. **Data Preprocessing**: Extracts and labels ECG segments.
4. **Data Splitting**: Distributes data across 5 clients.
5. **CNN Model Definition**: Builds the 1D CNN.
6. **FL Client Implementation**: Defines `ECGClient` for local training/evaluation.
7. **FL Simulation**: Runs 10 rounds of federated learning.
8. **Evaluation**: Tests the global model on a held-out set and plots an ECG segment.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script**:
   ```bash
   python federated_learning_arrhythmia_detection.py
   ```

## Results

### Data Shapes
- Test set: X shape: (11,428, 5), Y shape: (5,)
- Client data:
  - Client 1: X shape: (24,595, 720), Y shape: (24,595, 1)
  - Client 2: X shape: (22,976, 720), Y shape: (22,976, 1)
  - Client 3: X shape: (22,357, 720), Y shape:  (22,357, 1)
  - Client 4: X shape: (19,705, 720), Y shape: (19,705, 1)
  - Client 5: X shape: (19,785, 720), Y shape: (19,785, 1)

### Model Summary
```
Model: "sequential_1"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv1d_2 (Conv1D)               │ (None, 718, 32)        │           128 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d_2 (MaxPooling1D)  │ (None, 359, 32)        │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_3 (Conv1D)               │ (None, 357, 64)        │         6,208 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d_3 (MaxPooling1D)  │ (None, 178, 64)        │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_1 (Flatten)             │ (None, 11392)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 64)             │       729,152 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 735,553 (2.81 MB)
Trainable params: 735,553 (2.81 MB)
Non-trainable params: 0 (0.00 B)
```

### Federated Learning Performance
- **Simulation Duration**: 8,038.53 seconds (10 rounds).
- **Distributed Loss**:
  - Round 1: 0.2843
  - Round 2: 0.3028
  - Round 3: 0.2864
  - Round 4: 0.2531
  - Round 5: 0.2421
  - Round 6: 0.2379
  - Round 7: 0.2206
  - Round 8: 0.1959
  - Round 9: 0.1814
  - Round 10: 0.1871

### Test Set Evaluation
- **Test Loss**: 0.0710
- **Test Accuracy**: 0.9802
- **Classification Report**:
  ```
              precision    recall  f1-score   support
      Normal       0.98      0.99      0.98      7491
    Abnormal       0.98      0.96      0.97      3937
    accuracy                           0.98     11428
   macro avg       0.98      0.98      0.98     11428
weighted avg       0.98      0.98      0.98     11428

### Visualization
An example ECG segment from the test set is plotted with its label (normal/abnormal).

## Discussion

The federated learning approach achieved a test accuracy of 98.02%, with balanced precision and recall for both normal and abnormal classes. The decreasing loss trend over 10 rounds indicates effective model convergence. The high performance on the test set suggests that the global model generalizes well across unseen data, despite training on decentralized datasets. The privacy-preserving nature of FL makes this approach suitable for real-world healthcare applications.

## Future Improvements
- **Multi-Class Classification**: Extend to classify specific arrhythmia types.
- **Differential Privacy**: Add noise to weight updates for enhanced privacy.
- **Hyperparameter Optimization**: Tune CNN architecture and FL parameters.
- **Real-World Deployment**: Test on edge devices or hospital systems.
- **Scalability**: Increase the number of clients and dataset size.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- **MIT-BIH Arrhythmia Database**: Open-access ECG data.
- **Flower Framework**: Federated learning simulation.
- **TensorFlow & wfdb**: Model development and data processing.

