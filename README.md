# Human Movement Classifier (HAR) - PyTorch  

This repository contains a **Human Activity Recognition (HAR) model** using **PyTorch**.  
The model is trained to classify human movements based on **accelerometer data (x, y, z axes).**  

## 🚀 Features  
- **Multi-class classification** of activities (`sit`, `walk`, `stairsup`, `stairsdown`, `bike`).  
- **Deep Learning Model**: Fully connected neural network (FCNN).  
- **Dataset**: `Heterogeneous_accelerometer_HAR.csv` (Download below ⬇️).  
- **Evaluation**: Accuracy is calculated using a test dataset.  

## 📂 Dataset  
The dataset contains motion data from an accelerometer sensor with 3 axes (`x`, `y`, `z`).  
The labels (`gt`) represent the activity type.  

| x | y | z | gt |
|----|----|----|-------|
| 0.3 | -0.8 | 9.1 | sit |
| -0.5 | 0.2 | 9.5 | walk |
| 0.1 | -0.3 | 8.8 | stairsup |
| ... | ... | ... | ... |

📥 **Download Dataset**: [Heterogeneous_accelerometer_HAR.csv](PUT_YOUR_CSV_LINK_HERE)  

