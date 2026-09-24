# Glass Sheet Counting

This project presents a machine learning approach for predicting the number of glass sheets in a stack from images.

The problem has both a **classification and regression component**. The number of sheets is discrete and can be represented as categorical classes, while the relative dimensions of the glass stack provide a useful regression signal.

## Approach

The approach consists of:

* Custom CNN for learning image representations.
* Classification head for predicting the number of sheets as classes.
* Regression head for directly predicting the number of sheets.
* Image processing for extracting geometric features.
* Spectral analysis for generating additional predictive features.
* Data augmentation to improve robustness to changing lighting conditions.
* Multi-task learning using classification, regression, and consistency losses.

## Feature Engineering

I generated engineered features using image processing and spectral analysis.

Image processing is used to estimate the relative pixel width and position of the glass stack.

I also treat a selected image pixel row as a spectrum. The observed peaks are compared with theoretical peak positions for different numbers of sheets. This produces a spectral score indicating how well a given class fits the image.

The spectral baseline showed a stronger predictive signal than the random and majority-class baselines. Therefore, the spectral model is used as a **feature generator** for the main model rather than as the final predictor.

## Model

The main model is a custom CNN with a classification head and a regression head.

Both heads are conditioned on the CNN representation and engineered features, including:

* Relative pixel width of the glass stack
* Relative left, right, and center positions
* Spectral properties

## Learning Objective

The model uses four losses. The Huber and MAE losses train the regression head, cross-entropy trains the classification head, and the consistency loss encourages the regression head to use the classifier's prediction.

## Data

The data is automatically downloaded from Google Drive and curated into Pandas data frames. Images without labels are placed in the test set, while labeled data is split into training and validation data.

PyTorch dataset classes are then used to prepare the data for training.

## Model Interpretation

Feature maps from the CNN show that the model learned meaningful visual filters. In particular, the model learns vertical edges that are useful for identifying the boundaries between glass sheets.

## Assumptions

* Images have the same dimensions.
* Glass sheets have similar alignment or placement to the training data.
* The background is brighter than the glass stack.

## Tools

Python · PyTorch · Pandas · NumPy · OpenCV · Scikit-learn
