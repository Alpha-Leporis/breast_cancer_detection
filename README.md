# Breast Cancer Detection Using Vision Transformer

This project uses a Vision Transformer (ViT) model to detect breast cancer from ultrasound images. The dataset is sourced from Kaggle and includes three categories: benign, malignant, and normal.

## Project Directory Structure
```
breast_cancer_detection/
├── data/
│ ├── raw/
│ │ ├── benign/
│ │ ├── malignant/
│ │ ├── normal/
│ ├── processed/
│ ├── train/
│ │ ├── benign/
│ │ ├── malignant/
│ │ ├── normal/
│ ├── val/
│ ├── benign/
│ ├── malignant/
│ ├── normal/
├── src/
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
├── notebooks/
│ ├── breast_cancer_detection.ipynb
├── requirements.txt
├── README.md
└── run.py
```
## Setup Instructions
### 1. Clone the Repository

```bash
$ git clone https://github.com/Alpha-Leporis/breast_cancer_detection.git
$ cd breast_cancer_detection
```

### 2. Install Required Packages
Ensure you have Python 3.8 or higher installed. Install the required packages using:
```bash
$ pip install -r requirements.txt
```

### 3. Download the Dataset
Download the Breast Ultrasound Images Dataset from Kaggle [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

Extract the dataset into the data/raw directory.

### 4. Preprocess the Data
Run the data preprocessing script to organize the dataset into training and validation sets.
```bash
$ python src/data_preprocessing.py
```

### 5. Train the Model
Run the main script to train the Vision Transformer model.
```bash
$ python run.py
```

# File Descriptions
#### `data/`
* raw/: Contains the original dataset files as downloaded.
* processed/: Contains the organized dataset ready for training and validation, split into train and val directories.

#### `src/`
Contains all the Python scripts necessary for data processing, model definition, training, and evaluation.

* data_preprocessing.py : Script to preprocess and organize the dataset into the required directory structure.
* dataset.py: Script to handle data loading and transformations using PyTorch's DataLoader.
* model.py: Script to define and initialize the Vision Transformer model.
* train.py: Script to handle the training process of the model.
* evaluate.py: Script to evaluate the trained model on the validation set.

#### `requirements.txt`
A list of required Python packages.

#### `run.py`
A main script to run the complete training and evaluation pipeline.

#### `predict.py`
A script to load the trained model, preprocess the input image, make predictions, and print the predicted class.

# Acknowledgements
The dataset used in this project is provided by [Arya Shah](https://www.kaggle.com/aryashah2k) on [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
