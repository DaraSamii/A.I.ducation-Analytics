# A.I.ducation Analytics

## Table of Contents
- [Project Overview](#project-overview)
- [Data Pre-Processing](#data-pre-processing)
  - [Dataset](#dataset)
    - [Facial Expression Recognition (FER) 2013 Dataset](#facial-expression-recognition-fer-2013-dataset)
    - [FER+](#fer)
  - [Data Cleaning](#data-cleaning)
    - [Eliminating Images Labeled as 'Not Face'](#eliminating-images-labeled-as-not-face)
    - [Eliminating Images Labeled as 'Unknown'](#eliminating-images-labeled-as-unknown)
  - [Labeling](#labeling)
    - [Anger](#anger)
    - [Neutral](#neutral)
    - [Focused](#focused)
    - [Bored](#bored)
  - [Dataset Visualization](#dataset-visualization)

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

View the bar-plot of images per desired emotion in [./imgs/final_countplot.svg](./imgs/final_countplot.svg).

Sample images for each emotion are available in [./imgs/](https://github.com/DaraSamii/A.I.ducation-Analytics/tree/main/imgs).

*For more details, refer to the complete [Project Report](./Project_Report.pdf).*
