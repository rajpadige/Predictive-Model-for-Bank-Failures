# Predictive-Model-for-Bank-Failures
 In this project, bank failures were predicted using advanced ML algorithms and historical financial data of over 94,000 FDIC-insured banks. A dataset of 38 key variables from 1984 onwards was prepared using FDIC's BankFind Suite API. The model accurately identified most failed banks, demonstrating its effectiveness in predicting bank failures.


# Introduction

On March 10th, 2023, the banking industry took a huge hit when Silicon Valley Bank reported its failure. Two days after that, on March 12th, 2023, Signature Bank also reported its failure. These two bank failures are second and third on the list of the largest, the most prominent being the Washington Mutual Bank failure, which occurred during the 2008 crisis. These recent events beg the question of whether a domino effect is in its early stages and whether we could be staring at the proportions of the 2008 financial crisis.

The Federal Deposit Insurance Corporation is a United States government corporation supplying deposit insurance to American commercial and savings bank depositors. The FDIC preserves public confidence in our financial system by providing insurance for deposits up to $250,000 and monitoring risk for financial institutions, thus limiting the risk to the average depositor when a bank or savings institution fails.

Our study aims to develop a preemptive tool that can be used to predict the failure of a bank based on the objective financial indicators of a bank's historical performance. Regulators and individual banks may leverage such a tool in conjunction with other environmental and contextual factors to put preventive mechanisms in place if adverse scenarios are estimated.

We used various machine learning algorithms while building our model, which includes a selection of five diverse models for comparison. The models chosen are Random Forest, Decision Tree, Logistic Regression, Support Vector Machine, and k-Nearest Neighbors, all of which have different strengths in classifying binary outcomes. Random Forest produces an ensemble of decision trees to prevent overfitting, while Decision Tree generates a tree-like structure for interpretable predictions. Logistic Regression can handle multiple predictors, Support Vector Machine effectively handles linear and non-linear data, and the k-Nearest Neighbors model assigns new data to the majority class of its nearest neighbors. Through evaluating and comparing their performance, the study aims to determine the best model for predicting bank failures.

# Literature Review

In 2018, the European Banking Authority (EBA) conducted a study exploring machine learning techniques for predicting bank insolvencies. The EBA study used data over eight years from 2008 to 2016 and tested different machine learning models, like logistic Regression, decision trees, random forests, and artificial neural networks.

Another study by researchers from Bryant University published in the Journal of Economics and Finance (2018) looked to predict bank failures specifically in the United States using similar machine learning models to the EBA, such as decision trees and random forests along with support vector machines. The data used was over a 10-year period from 2005 - 2015. A student from the University of Brasilia (2021) conducted a similar study to predict bank failures in Brazil using eight-year data from 2012 - 2020, using all the mentioned models along with logistic Regression and neural networks.

In an analysis of studies published in the Journal of Applied Finance and Banking (2021), researchers looked to identify the best models for predicting corporate bankruptcy. While not bank failure, the structure and methods of the studies served as inspiration.

Ong, Loo, Wong, and Ong (2020) compared the use of machine learning and traditional techniques in predicting bank distress in the Association of Southeast Asian Nations (ASEAN-5) countries, including Indonesia, Malaysia, Thailand, Vietnam, and the Philippines.

Their comparison found that machine-learning models typically outperformed traditional techniques regarding overall accuracy and recall rates. The random forest model had the highest accuracy in predicting bank failures in most of the conducted studies. All the mentioned studies helped inspire our own study as we look to find the most accurate model for predicting bank failure using various methods and ensembles of methods.

# Problem Statement Definition

The question that we seek to answer through our study is:

*Can we predict the failure of a bank from historical data on the objective financial ratios that indicate a bank's performance and health?*

We want to investigate financial measures that are publicly available and have a high degree of comparability across the banking sector. We understand that several other factors may contribute to a bank's failure to a greater extent. Such factors include regulatory supervision, macroeconomic scenario, natural disasters, geographical distribution, and reach. However, we are not attempting to control these contextual and environmental determinants. Our question, thus, may be read as – *can we predict a bank's failure just by studying the trends in the financial metrics of the banks that have failed previously?*

Against this presumption, we define our goal statement as:

*We'd consider our model a success if we are able to identify most banks that have failed accurately.*

# Methodology Adopted

Our problem statement is inherently a classification problem. And we want to tackle it with advanced machine-learning algorithms at our disposal. We have selected five different machine learning techniques to predict bank failures. These were carefully selected to encompass diverse classification techniques, thereby enabling a comprehensive comparison of performances across different models. The models that we selected are Random Forest, Decision Tree, Logistic Regression, Support Vector Machine, and k-Nearest Neighbors.

These models have different strengths when it comes to classifying binary outcomes. For instance, Random Forest provides an ensemble of decision trees, thus preventing overfitting. Decision Tree generates a tree-like structure for interpretable predictions. Logistic Regression can handle multiple predictors efficiently. Support Vector Machine handles linear and non-linear data effectively. And k-Nearest Neighbors assigns new data points to the majority class of its nearest neighbors.

The performance of each model will be measured based on various metrics such as accuracy, sensitivity, specificity, precision, and F1-score. Additionally, we will create an ensemble model to observe if a combination of these models improves overall performance.

# Data Collection and Preparation

To build and train our model, we collected financial data from the Federal Deposit Insurance Corporation's BankFind Suite API. We collected data for operational banks and failed banks and identified our response variables as 1 (failed) and 0 (operational). We obtained data from 1984 to 2022 to ensure a substantial dataset for training and testing our models.

The financial ratios we used include Return on Assets (ROA), Net Interest Margin (NIM), Efficiency Ratio (ER), Loan to Assets (LTA), and Loan Loss Provision to Total Assets (LLP).

To ensure the quality and reliability of our data, we performed data cleaning, including handling missing values and duplicates. We also addressed issues related to multicollinearity to avoid overfitting.

For the analysis, we considered yearly records for banks, which allowed for better interpretability and a clearer understanding of trends.

# Analysis and Results

We split our dataset into training and testing sets. The training set comprises 80% of the data, and the testing set comprises 20%.

We trained each of the five models on the training set and evaluated their performance on the testing set. The evaluation metrics used are accuracy, sensitivity, specificity, precision, and F1-score.

The results of our analysis are as follows:

- **Random Forest:**
  - Accuracy: 96.51%
  - Sensitivity: 95.98%
  - Specificity: 96.73%
  - Precision: 96.33%
  - F1-score: 96.16%

- **Decision Tree:**
  - Accuracy: 94.82%
  - Sensitivity: 93.74%
  - Specificity: 95.05%
  - Precision: 94.56%
  - F1-score: 94.15%

- **Logistic Regression:**
  - Accuracy: 91.86%
  - Sensitivity: 87.77%
  - Specificity: 93.53%
  - Precision: 90.82%
  - F1-score: 89.27%

- **Support Vector Machine:**
  - Accuracy: 93.91%
  - Sensitivity: 92.55%
  - Specificity: 94.18%
  - Precision: 93.71%
  - F1-score: 93.13%

- **k-Nearest Neighbors:**
  - Accuracy: 95.03%
  - Sensitivity: 93.57%
  - Specificity: 95.41%
  - Precision: 94.90%
  - F1-score: 94.23%

# Conclusion and Recommendations

Based on the results of our analysis, we found that the Random Forest model outperformed the other models, achieving the highest accuracy of 96.51% and the highest sensitivity of 95.98% in predicting bank failures.

However, to further improve the predictive capabilities and robustness of our model, we created an ensemble model that combines the predictions of all five models. The ensemble model achieved an accuracy of 96.29% and a sensitivity of 95.69%, making it a strong candidate for predicting bank failures.

We recommend using the ensemble model as a preemptive tool to detect early signs of bank distress. Regulators and banks can utilize these predictions to implement preventive measures and mitigate potential risks to stabilize the financial system.

In conclusion, our study demonstrates that machine learning models can be effective tools for predicting bank failures based on historical financial indicators. However, it is essential to remember that this is just one aspect of a comprehensive risk management strategy, and the model should be used in conjunction with other factors to make informed decisions.
