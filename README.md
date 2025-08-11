Uber Surge Pricing Prediction - Classification Project
--------------------------------------------------

**Project Description**
This project predicts Uber surge pricing categories based on various features such as location, weather conditions, and other contextual variables. 
We classify the Surge multipliers  into three categories:
- low 
- medium
- High

**Dataset**
- The dataset should be saved as `uber_surge_data.csv` in the same directory as the code.
- Required columns may include:
  - location
  - weather
  - surge_multiplier
  - other contextual features (time, distance, etc.)

**Methodology**
1. Data preprocessing: Label encoding for categorical features.
2. Handling imbalance: Random oversampling to balance the target categories.
3. Model: Logistic Regression classifier is used to classify surge categories.
4. Evaluation: Accuracy, weighted F1 score, classification report, and confusion matrix.

**Dependencies**
- pandas
- scikit-learn
- imbalanced-learn

**How to Run**
1. Install dependencies:
   pip install pandas scikit-learn imbalanced-learn
2. Place the dataset `uber_surge_data.csv` in the project folder.
3. Run the Python script.

**Output**
- Console output showing accuracy, F1 score, classification report, and confusion matrix.

--------------------------------------------------
Author: Md Hidayat Ali Ansari
Date: 2025
