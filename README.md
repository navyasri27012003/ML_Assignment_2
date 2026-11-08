# Machine Learning Assignment 2
## Bank Marketing Classification

### 1. About the Assignment

For this assignment, I selected the Bank Marketing dataset from the UCI Machine Learning Repository.

The main objective is to predict whether a customer will subscribe to a term deposit based on the information available in the dataset.

I implemented five machine learning classification algorithms and compared their performance using different evaluation metrics.

The five models I used are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

I also developed a Streamlit application where the user can upload test data, select a model, and check its performance.

### 2. Project Links

**GitHub Repository:**

https://github.com/navyasri27012003/ML_Assignment_2

**Streamlit Application:**

https://mlassignment2-nh4tbz6ltwauh8pg72dkhu.streamlit.app

### 3. Dataset Details

I used the Bank Marketing dataset available on the UCI website.

Dataset link: https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains information about bank customers and marketing campaigns.

**Dataset information:**

- Dataset file: `bank-full.csv`
- Total records: 45,211
- Number of input features: 16
- Target column: `y`
- Target values: `yes` and `no`

Here, `yes` means the customer subscribed to a term deposit, and `no` means the customer did not subscribe.

The dataset contains both numerical and categorical features such as age, job, marital status, education, balance, housing loan, personal loan, contact type, and campaign details.

### 4. Data Preprocessing

Before training the models, I performed the following steps:

1. Loaded the dataset using Pandas.
2. Checked the number of rows, columns, missing values, and target distribution.
3. Separated the input features and the target column.
4. Converted the target values into numerical values (`no = 0`, `yes = 1`).
5. Used `StandardScaler` for numerical columns.
6. Used `OneHotEncoder` for categorical columns.
7. Split the dataset into 80% training data and 20% testing data.
8. Used stratified splitting to maintain the target class distribution.

After splitting the dataset:

- Training records: 36,168
- Testing records: 9,043

I used the same training and testing data for all five models so that I could compare their performance.

### 5. Model Implementation

**Logistic Regression**

I implemented Logistic Regression as the first classification model. I used it to predict whether a customer would subscribe to a term deposit.

**Decision Tree**

I trained a Decision Tree classifier with a maximum depth of 10 and minimum leaf size of 10.

**K-Nearest Neighbors**

For KNN, I used 5 nearest neighbors with distance-based weights.

**Naive Bayes**

I implemented Gaussian Naive Bayes. I used a dense array conversion step in the pipeline for this model.

**Support Vector Machine**

I implemented SVM using `LinearSVC`. I used the decision function scores to calculate AUC.

After training, I saved the models using Joblib so that they could be loaded into the Streamlit application without training them again.

### 6. Model Evaluation

I evaluated all five models using these six metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

I also generated confusion matrices, classification reports, ROC curves, and comparison graphs to understand the performance of each model.

### 7. Model Comparison

The following results were obtained after testing all five models on the same test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8993 | 0.8799 | 0.6031 | 0.4064 | 0.4856 | 0.4425 |
| KNN | 0.8965 | 0.8293 | 0.5993 | 0.3478 | 0.4402 | 0.4050 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| SVM | 0.9001 | 0.9072 | 0.6605 | 0.3015 | 0.4140 | 0.4016 |

### 8. Observations

**Logistic Regression**

Logistic Regression gave the highest accuracy of 90.12%. Its AUC was also good, but the recall was low. This means the model was not able to identify many of the customers who actually subscribed.

**Decision Tree**

Decision Tree gave an accuracy of 89.93%. It achieved the highest F1 Score and MCC compared to the other models. I observed that it gave a better balance between precision and recall.

**K-Nearest Neighbors**

KNN gave an accuracy of 89.65%. Its performance was close to the other models in terms of accuracy, but its AUC and F1 Score were comparatively lower.

**Naive Bayes**

Naive Bayes gave the lowest accuracy of 85.48%, but it achieved the highest recall of 51.98%. It identified more actual subscribers, although it also predicted more customers incorrectly as subscribers.

**Support Vector Machine**

SVM gave an accuracy of 90.01%. It achieved the highest AUC and precision among all five models. However, its recall was lower than the other models.

### 9. Best Model

After comparing the results, I observed that each model performed better in different metrics.

Logistic Regression gave the highest accuracy, SVM gave the highest AUC and precision, and Naive Bayes gave the highest recall.

For this dataset, I considered **Decision Tree** as the better overall model because it achieved the highest F1 Score and MCC.

Since the dataset contains more `no` records than `yes` records, I felt that comparing only accuracy would not be sufficient.

### 10. Streamlit Application

After completing the model training and evaluation, I created a Streamlit application to test the saved models.

In the application, I added a dropdown to select any one of the five classification models.

I also added an option to upload a CSV file containing the test data.

Once the file is uploaded and a model is selected, the application displays:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion Matrix
- Classification Report

I tested the application by uploading `test_data.csv` and selecting each of the five models.

The application displayed the evaluation results for all five models.

**Live Application:**

https://mlassignment2-nh4tbz6ltwauh8pg72dkhu.streamlit.app

### 11. How to Use the Application

1. Open the Streamlit application link.
2. Select a classification model from the dropdown.
3. Upload the `test_data.csv` file.
4. Check the evaluation metrics displayed on the screen.
5. Scroll down to view the confusion matrix and classification report.
6. Select another model to compare the results.

The uploaded CSV should contain the input features and the target column `y` with `yes` or `no` values.

### 12. Files in the GitHub Repository

| File / Folder | Description |
|---|---|
| `ML_Assignment_2.ipynb` | Jupyter notebook containing model implementation and evaluation |
| `app.py` | Python code for the Streamlit application |
| `requirements.txt` | Python libraries required to run the application |
| `test_data.csv` | Test dataset used for evaluating the models |
| `model/` | Saved machine learning models |
| `README.md` | Project details, results, and observations |

### 13. Tools and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook
- Google Colab
- Streamlit
- GitHub

### 14. Conclusion

In this assignment, I implemented five classification algorithms using the Bank Marketing dataset.

I trained all five models using the same dataset and evaluated them using six different metrics.

From the results, I observed that Logistic Regression gave the highest accuracy, SVM gave the highest AUC and precision, Naive Bayes gave the highest recall, and Decision Tree gave the highest F1 Score and MCC.

Based on the results, I selected Decision Tree as the overall better model for this dataset.

I also created and deployed a Streamlit application where the trained models can be tested by uploading a CSV file.

Through this assignment, I understood how preprocessing, model training, evaluation metrics, and model comparison are performed for a classification problem.

### 15. References

1. UCI Machine Learning Repository – Bank Marketing Dataset  
   https://archive.ics.uci.edu/dataset/222/bank+marketing

2. Scikit-learn Documentation  
   https://scikit-learn.org/stable/

3. Streamlit Documentation  
   https://docs.streamlit.io/
