# A.I.ducation Analytics

## Project Overview

This project, undertaken as part of COMP 6721 Applied Artificial Intelligence, explores the realm of facial expression recognition. The primary objective is to develop an understanding of emotions depicted in facial images and create a robust model for emotion recognition. The project focuses on preprocessing a comprehensive dataset, combining FER2013 and FER+, and enhancing it to facilitate accurate training and evaluation of emotion recognition models.

## Data Pre-Processing

### Dataset

#### Facial Expression Recognition (FER) 2013 Dataset

The primary dataset utilized is the FER2013 dataset, a pivotal resource in facial emotion recognition. Comprising grayscale images measuring $48 \times 48$ pixels, FER2013 contains 32,298 samples categorized into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The dataset is available on [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

#### FER+

FER+ is an enhanced version of FER2013, incorporating scores assigned by ten taggers to each image. The dataset includes additional labels such as 'Not Face' and 'Unknown.' Merging FER+ with FER2013 yields an accurate dataset where emotion intensity is gauged by tagger votes. The dataset is available on [GitHub](https://github.com/Microsoft/FERPlus).

### Data Cleaning

#### Eliminating Images Labeled as 'Not Face'

Images labeled as 'Not Face' were removed from the dataset, given their limited count compared to the overall dataset size.

#### Eliminating Images Labeled as 'Unknown'

Images labeled as 'Unknown' were selectively removed based on the score assigned. Images with a score of 5 or greater were excluded to maintain dataset coherence.

### Labeling

Images were labeled into four categories: Anger, Neutral, Focused, and Bored. The labeling criteria are detailed as follows:

- **Anger:** Image labeled as 'Angry' if the anger score surpasses other emotions or is at least 2.
- **Neutral:** Image labeled as 'Neutral' if the neutral score surpasses other emotions or is greater than 6.
- **Focused:** Image labeled as 'Focused' if sadness and anger scores are zero, and the neutral score is higher than other emotions.
- **Bored:** Image labeled as 'Bored' if happiness and fear scores are zero, sadness and neutral scores are non-zero, and the anger score + 2 is less than the sadness score.

### Dataset Visualization

After cleaning and labeling, the dataset comprises:
- Neutral: 3789 samples
- Angry: 3954 samples
- Focused: 4553 samples
- Bored: 3960 samples

View the bar-plot of images per desired emotion in 

![coutplot](./LaTeX/imgs/final_countplot.svg).

## Training Models
In Section 1, we created a dataset consisting of 14,831 samples. The data split ratios were chosen as 70%, 15%, and 15% for the training, validation, and test sets, respectively. This resulted in 10,382 samples for training, 2,225 samples for validation, and 2,224 samples for testing.

### Augmentation
To enhance the diversity of our training data and improve the robustness of the model, we applied two aug- mentation techniques during training:
 - Horizontal Flip with a probability of 0.5
 - Random Rotation between −10 and 10 degrees

### CNN architecture
Implementation of the models is conducted in the PyTorch library. 12 distinct CNN-based models are introduced to proficiently categorize images into 4 classes.

![model](./LaTeX/imgs/resnet.svg)

### Hyper-parameters

| Parameter |	Values | 
| --- | --- | 
| Learning Rate | 0.001 |  
| Epochs | 	100 | 
| Batch Size | 	64 | 
| Optimizer | 	Adam | 
| Loss Function | Cross-Entropy|  
| Weight Decay | 	0.0001 | 

### Training Process
After creating a Training Data Loader, Validation DataLoader and Test Data Loader, a Learner class was created with couples the data loader and model. the learner handles traning of the model, logging the metrics, and loss and saves and load the trained models specified. Each model was trained by train dataset an epoch, followed by evaluted the model with validation dataset and if a new accuracy was found the model would be saved.

![training](./LaTeX/imgs/train.svg)

### Hyper-paremeter’s effect

*For more details, refer to the complete [Project Report](./Project_Report.pdf).*
