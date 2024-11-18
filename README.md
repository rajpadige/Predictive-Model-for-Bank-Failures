Predictive Model for Bank Failures 💼📉
Introduction 🌟
In March 2023, the financial sector faced significant challenges as Silicon Valley Bank and Signature Bank failed, raising concerns about a potential ripple effect reminiscent of the 2008 financial crisis. To address these concerns, we developed a predictive model that utilizes historical financial data and machine learning algorithms to identify early signs of bank distress.

Our goal was to create a tool that accurately predicts bank failures based on objective financial indicators. This tool aims to empower regulators and banks with the ability to detect potential risks early and take preventive action to safeguard the financial system.

Literature Review 📚🔍
Drawing inspiration from studies conducted by organizations like the European Banking Authority and academic institutions such as Bryant University and the University of Brasilia, we explored various machine learning techniques. These studies demonstrated the potential of models like Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines in predicting bank failures in diverse regions.

Problem Statement 🎯
Our research aimed to answer a critical question: Can we predict bank failures using historical financial data and performance metrics? We focused on publicly available measures that ensure comparability across banks, seeking patterns that indicate potential risks.

Methodology 📊🔬
We employed five machine learning techniques to tackle this binary classification problem:

Random Forest
Decision Tree
Logistic Regression
Support Vector Machine (SVM)
k-Nearest Neighbors (k-NN)
Each model’s unique strengths made it a valuable candidate for our analysis.

Data Collection and Preparation 📈🔍
The dataset included historical financial data from over 94,000 FDIC-insured banks, spanning from 1984 to the present. Data was sourced using FDIC's BankFind Suite API and included financial metrics like:

Return on Assets (ROA)
Net Interest Margin (NIM)
Efficiency Ratio (ER)
Loan to Assets (LTA)
Loan Loss Provision to Total Assets (LLP)
Rigorous data cleaning and handling of missing values ensured a high-quality dataset. The data was split into training and testing sets to evaluate model performance effectively.

Analysis and Results 📝📈
The models were evaluated using metrics such as accuracy, sensitivity, specificity, precision, and F1-score. The performance results were as follows:

Random Forest:

Accuracy: 96.51%
Sensitivity: 95.98%
Specificity: 96.73%
Precision: 96.33%
F1-score: 96.16%
Decision Tree:

Accuracy: 94.82%
Sensitivity: 93.74%
Specificity: 95.05%
Precision: 94.56%
F1-score: 94.15%
Logistic Regression:

Accuracy: 91.86%
Sensitivity: 87.77%
Specificity: 93.53%
Precision: 90.82%
F1-score: 89.27%
Support Vector Machine (SVM):

Accuracy: 93.91%
Sensitivity: 92.55%
Specificity: 94.18%
Precision: 93.71%
F1-score: 93.13%
k-Nearest Neighbors (k-NN):

Accuracy: 95.03%
Sensitivity: 93.57%
Specificity: 95.41%
Precision: 94.90%
F1-score: 94.23%
Conclusion and Recommendations 🎉🔮
The Random Forest model demonstrated the highest accuracy at 96.51% and sensitivity at 95.98%, making it the most effective for predicting bank failures. To further enhance predictive power, we recommend employing an ensemble model that combines the strengths of all five techniques. This approach achieves an overall accuracy of 96.29% and sensitivity of 95.69%.

This model provides regulators and banks with a robust tool for early risk identification. While powerful, it should be complemented with contextual factors to ensure well-rounded decision-making and risk management.

By leveraging this predictive tool, we can contribute to a more resilient and stable financial future.
