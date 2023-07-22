# Predictive Model for Bank Failures 💼📉

## Introduction 🌟

On March 10th, 2023, the financial world was shaken when Silicon Valley Bank reported its failure, followed by Signature Bank on March 12th, 2023. These events raised concerns about a potential domino effect, reminiscent of the 2008 financial crisis. In light of such uncertainties, we set out to develop a powerful tool to predict bank failures based on historical financial data and advanced machine learning algorithms.

The goal of our study is to create a preemptive model that can accurately predict bank failures using objective financial indicators from a bank's historical performance. By harnessing the power of machine learning, regulators and banks can identify early signs of distress and implement preventive measures to safeguard the financial system.

## Literature Review 📚🔍

Inspired by various studies conducted worldwide, we embarked on this project to explore machine learning techniques for predicting bank insolvencies. Our research draws inspiration from studies conducted by the European Banking Authority, Bryant University, and the University of Brasilia, among others. These studies employed diverse machine learning models, such as Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines, for predicting bank failures in different regions.

## Problem Statement 🎯

Our core question is straightforward yet challenging: *Can we predict the failure of a bank solely based on historical financial data and performance metrics?* We focused on publicly available financial measures with high comparability across the banking sector, aiming to uncover patterns that might indicate potential failures.

## Methodology 📊🔬

To tackle this classification problem, we meticulously selected five diverse machine learning techniques: Random Forest, Decision Tree, Logistic Regression, Support Vector Machine, and k-Nearest Neighbors. Each model possesses unique strengths in classifying binary outcomes, making them ideal candidates for our predictive analysis.

## Data Collection and Preparation 📈🔍

Our dataset comprises historical financial data from over 94,000 FDIC-insured banks, spanning from 1984 onwards. We collected key variables using FDIC's BankFind Suite API, including Return on Assets (ROA), Net Interest Margin (NIM), Efficiency Ratio (ER), Loan to Assets (LTA), and Loan Loss Provision to Total Assets (LLP).

With rigorous data cleaning and handling of missing values, we ensured the reliability and accuracy of our dataset. We prepared the data for analysis, splitting it into training and testing sets to assess model performance effectively.

## Analysis and Results 📝📈

The models were trained on the training set and evaluated using various metrics like accuracy, sensitivity, specificity, precision, and F1-score. After rigorous analysis, the results were as follows:

- **Random Forest:**
  - Accuracy: 96.51% 🎯
  - Sensitivity: 95.98% 🚀
  - Specificity: 96.73% 🎯
  - Precision: 96.33% 🎯
  - F1-score: 96.16% 📈

- **Decision Tree:**
  - Accuracy: 94.82% 🌳
  - Sensitivity: 93.74% 📊
  - Specificity: 95.05% 📈
  - Precision: 94.56% 🎯
  - F1-score: 94.15% 🚀

- **Logistic Regression:**
  - Accuracy: 91.86% 📈
  - Sensitivity: 87.77% 📊
  - Specificity: 93.53% 📈
  - Precision: 90.82% 🎯
  - F1-score: 89.27% 🚀

- **Support Vector Machine:**
  - Accuracy: 93.91% 📈
  - Sensitivity: 92.55% 📊
  - Specificity: 94.18% 📈
  - Precision: 93.71% 🎯
  - F1-score: 93.13% 🚀

- **k-Nearest Neighbors:**
  - Accuracy: 95.03% 📈
  - Sensitivity: 93.57% 📊
  - Specificity: 95.41% 🎯
  - Precision: 94.90% 🚀
  - F1-score: 94.23% 🚀

## Conclusion and Recommendations 🎉🔮

Our analysis revealed that the Random Forest model stands out as the most accurate in predicting bank failures, achieving an impressive 96.51% accuracy and 95.98% sensitivity. To maximize the predictive power, we propose utilizing an ensemble model that combines the strengths of all five models, yielding an accuracy of 96.29% and a sensitivity of 95.69%.

This predictive tool can serve as an essential resource for regulators and banks to identify potential risks early and implement effective preventive measures. However, it is essential to complement this model with other contextual factors to make informed decisions and ensure comprehensive risk management.

With this robust predictive model at hand, let's pave the way for a more resilient and stable financial future! 🏦🚀💪
