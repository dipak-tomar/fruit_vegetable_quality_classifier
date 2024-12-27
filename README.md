# Fruit and Vegetable Quality Classification

## Description
This project uses an ANN model to classify fruits and vegetables into:
- Export Quality
- Domestic Market Quality
- Rejected

## Folder Structure
- `resized_dataset/`: Contains the dataset organized by quality and produce type.
- `src/`: Python scripts for preprocessing, training, and inference.
- `models/`: Saved trained model files.
- `outputs/`: Outputs such as graphs and results.


## Instructions
1. Preprocess the dataset:
   ```bash
   python src/data_preprocessing.py
   ```
2. Train the model:
   ```bash
   python src/model_training.py
   ```
3. Run inference:
   ```bash
   streamlit run src/model_inference.py
   ```
