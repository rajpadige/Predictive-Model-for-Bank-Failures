library(tidyverse)
library(utils)
library(dplyr)

# Load the data
bank_data <- read.csv("bankdata.csv")

#DATA CLEANING 

# Find columns with 0 values
zero_cols <- colSums(bank_data == 0)
#Determined the variables which had a lot of holes which 
#would only give damage to our model  or lead to training a 
#false model because of not having data 
# Print the number of 0 values in each column
print(zero_cols)

# Remove the irrelevant columns and fill missing values with 0
bank_data_clean <- bank_data[, -c(1,2,3,4,5,6,7,8,14,21,22,25,37,38)]
bank_data_clean[is.na(bank_data_clean)] <- 0

# Scale the numerical columns
bank_data_clean <- cbind(scale(bank_data_clean[, 1:(ncol(bank_data_clean) - 1)]), bank_data_clean[, ncol(bank_data_clean)])

bank_data_clean <- data.frame(bank_data_clean)

# Change the column name 'V27' to 'FAILED'
colnames(bank_data_clean)[colnames(bank_data_clean) == "V27"] <- "FAILED"

# Reattach CERT and convert FAILED variable to a factor
cert <- bank_data$CERT
bank_data_clean$CERT <- cert

# Split the dataset into training and test sets based on the CERT variable
x <- bank_data_clean %>% distinct(CERT)
x <- x$CERT
train_banks <- sample(x, 0.8 * length(x))
train_data <- bank_data_clean %>% filter(CERT %in% train_banks)
test_data <- bank_data_clean %>% filter(!CERT %in% train_banks)

# Remove CERT column from both datasets
train_data <- train_data %>% select(-CERT)
test_data <- test_data %>% select(-CERT)


#FEATURE SELECTION 

# Calculate the correlation matrix for all columns in train_data
corr_all <- cor(train_data)
corr_rank_all <- sort(abs(corr_all[, ncol(corr_all)]), decreasing = TRUE)

#FACTORING TRAIN FAILED
failedtrain <- train_data$FAILED  
failedtrain <- as.factor(failedtrain)

#FACTORING TEST FAILED
failedtest <- test_data$FAILED  
failedtest <- as.factor(failedtest)

# Remove FAILED column from both datasets
train_data <- train_data %>% select(-FAILED)
test_data <- test_data %>% select(-FAILED)

#Adding back factored FAILED COLUMNS
train_data$FAILED <- failedtrain
test_data$FAILED <- failedtest

str(train_data)
str(test_data)

View(corr_rank_all)

#SELECTED VARIABLED BASED ON CORRELATION ANALYSIS SO THERE IS NO ISSUE OF MULTICOLLINEARITY 
#PERFORMED VIF ANALYSIS TO HANDLE MULTICOLLINEARITY AND REMOVE VARIABLES 
#SELECTED 14 VARIABLES

#Getting Top required features 
selected_features <- names(corr_rank_all[c("INTEXPY", "NPERFV", "NOIJY", "ERNASTR", "LNLSNTV", "EQV", "NIMY", "ASTEMPM", "NONIXAY", "LNATRESR", "EEFFR", "LNRESNCR", "IDLNCORR", "NONIIAY")])
View(selected_features)

# Create a new data frame with the selected 14 features, excluding the FAILED variable
train_data_new <- train_data[, c(selected_features, "FAILED")]

# Create a new data frame with the selected features, excluding the FAILED variable
train_data_new <- train_data[, c(selected_features, "FAILED")]
# Convert the FAILED variable in the test data frame to a factor with the same levels as in the train data frame
test_data$FAILED <- factor(test_data$FAILED, levels = levels(train_data$FAILED))

#MODEL BUILDING 

# Make predictions on the test data
test_data_new <- test_data[, selected_features]

library(rpart)
library(rpart.plot)
library(Matrix)
library(e1071)
library(class)
library(randomForest)
library(caret)
library(glmnet)

#BEST PARAMETER FINDER 
#CrossValidation TrainControl 
control <- trainControl(method = "cv", number = 10)
rf_grid <- expand.grid(mtry = seq(2, 10, 2))
dt_grid <- expand.grid(cp = seq(0.01, 0.1, 0.01))
logreg_grid <- expand.grid(alpha = 0:1, lambda = seq(0.001, 0.1, 0.001))
svm_grid <- expand.grid(C = 2^(seq(-5, 5, 1)), sigma = 2^(seq(-5, 5, 1)))
knn_grid <- expand.grid(k = seq(1, 21, 2))

# Train a random forest model using the selected features and paramters grid
rf_model <- train(FAILED ~ ., data = train_data_new, method = "rf", trControl = control, tuneGrid = rf_grid, importance = TRUE)

# Train a decision tree model using the selected features and paramters grid
dt_model <- train(FAILED ~ ., data = train_data_new, method = "rpart", trControl = control, tuneGrid = dt_grid)

# Train a logistic regression model using the selected features and paramters grid
logreg_model <- train(FAILED ~ ., data = train_data_new, method = "glmnet", trControl = control, tuneGrid = logreg_grid)

# Train an SVM model using the selected features and paramters grid
svm_model <- train(FAILED ~ ., data = train_data_new, method = "svmRadial", trControl = control, tuneGrid = svm_grid, scale = FALSE)


# Train a k-NN model using the selected features and paramter grid
knn_model <- train(FAILED ~ ., data = train_data_new, method = "knn", trControl = control, tuneGrid = knn_grid)

#BEST PARAMS
print(rf_model$bestTune)
print(dt_model$bestTune)
print(logreg_model$bestTune)
print(svm_model$bestTune)
print(knn_model$bestTune)


# Display the variable importance plot
varImpPlot(rf_model)
#Random Forest Model with best parameters
rf_model_tuned <- randomForest(FAILED ~ ., data = train_data_new, ntree = 500, mtry = 2, importance = TRUE)
#Decision Tree Model  with best parameter
dt_model_tuned <- rpart(FAILED ~ ., data = train_data_new, method = "class", control = rpart.control(cp = 0.01))
# Plot the Decision Tree with best parameter
library(rpart.plot)
rpart.plot(dt_model_tuned)
#Logistic Regression Model with best parameter
logreg_model_tuned <- glmnet(x = model.matrix(FAILED ~ ., data = train_data_new), y = train_data_new$FAILED, family = "binomial", alpha = 1, lambda = 0.001)
#SVM Model with best parameter
svm_model_tuned <- svm(FAILED ~ ., data = train_data_new, kernel = "radial", cost = 4, gamma = 0.5, scale = FALSE)
#KNN Model with best parameter
test_data_failed <- test_data[, c(selected_features, "FAILED")]
# Train the k-NN model with best parameter
knn_model_tuned <- knn(train_data_new[, selected_features], test_data_new, train_data_new$FAILED, k = 1)


#PREDICTIONS

# Make predictions on the test data for each model
rf_predictions <- predict(rf_model_tuned, newdata = test_data_new)
dt_predictions <- predict(dt_model_tuned, newdata = test_data_new, type = "class")

# Convert the test dataset to a model matrix
test_data_matrix <- model.matrix(FAILED ~ ., data = test_data_failed)
# Get the predictions using the model matrix
logreg_predictions <- predict(logreg_model_tuned, newx = test_data_matrix, type = "response")
logreg_predictions <- ifelse(logreg_predictions > 0.5, 1, 0) # Convert probabilities to binary predictions
svm_predictions <- predict(svm_model_tuned, newdata = test_data_new)
knn_predictions <- knn_model_tuned

#PERFORMANCE EVALUATION
 # Evaluate the performance of each model
rf_cm <- confusionMatrix(rf_predictions, test_data$FAILED)
dt_cm <- confusionMatrix(dt_predictions, test_data$FAILED)

#-------------------------------
# Convert 'logreg_predictions' and 'test_data$FAILED' to factors with the same levels
logreg_predictions_factor <- factor(logreg_predictions, levels = levels(test_data$FAILED))
test_data_failed_factor <- factor(test_data$FAILED, levels = levels(test_data$FAILED))

# Calculate the confusion matrix
logreg_cm <- confusionMatrix(logreg_predictions_factor, test_data_failed_factor)
#-----------------------------------------
svm_cm <- confusionMatrix(svm_predictions, test_data$FAILED)
knn_cm <- confusionMatrix(knn_predictions, test_data$FAILED)


# Print the accuracy and other metrics for each model
cat("Random Forest Model:\n")
print(rf_cm)
cat("Decision Tree Model:\n")
print(dt_cm)
# Plot the decision tree
rpart.plot(dt_model_tuned)
#---------------------------------------------------------------------------------------------
# Print the accuracy and other metrics for the logistic regression model
cat("Logistic Regression Model:\n")
print(logreg_cm)
#----------------------------------------------------------------------------------------------

cat("SVM Model:\n")
print(svm_cm)
cat("k-NN Model:\n")
print(knn_cm)


# Create a data frame to store the accuracies
accuracy_table <- data.frame(Model = c("Random Forest", "Decision Tree","knn", "Log Reg", "SVM"),
                             Accuracy = c(as.numeric(rf_cm$overall["Accuracy"]),
                                          as.numeric(dt_cm$overall["Accuracy"]),
                                          as.numeric(knn_cm$overall["Accuracy"]),
                                          as.numeric(logreg_cm$overall["Accuracy"]),
                                          as.numeric(svm_cm$overall["Accuracy"])))

# Print the table
print(accuracy_table)
library(ipred)
# Train the ensemble model using bagging
ensemble_model <- bagging(FAILED ~ ., data = train_data_new,
                          coob = TRUE,
                          nbagg = 10,
                          predictors = list(rf_model, dt_model, logreg_model, svm_model, knn_model))

# Make predictions on the test data
ensemble_predictions <- predict(ensemble_model, newdata = test_data_new)

# Evaluate the performance of the ensemble model
ensemble_cm <- confusionMatrix(ensemble_predictions, test_data$FAILED)
ensemble_accuracy <- as.numeric(ensemble_cm$overall["Accuracy"])

# Add the ensemble accuracy to the table
accuracy_table <- rbind(accuracy_table, data.frame(Model = "Ensemble", Accuracy = ensemble_accuracy))
accuracy_table$Accuracy <- accuracy_table$Accuracy*100

# Print the updated table
print(accuracy_table)

# Evaluate the performance of the ensemble model
confusionMatrix(ensemble_predictions, test_data$FAILED)

# Load PrimeTest dataset
load("C:/Users/Ritvik Raj/Downloads/prime_test.RData")
ptest <- prime_test

# Remove the irrelevant columns and fill missing values with 0
ptest_clean <- ptest[, -c(1,2,3,4,5,6,7,8,38,41)]
ptest_clean[is.na(ptest_clean)] <- 0

# Scale the numerical columns
ptest_final<- scale(ptest_clean)
ptest_final<- data.frame(ptest_final)

# Reattach FAILED column and convert it to a factor
failed <- ptest$FAILED  
failed <- as.factor(failed)
ptest_final$FAILED <- failed

# Make predictions on the PrimeTest dataset using the trained random forest model
ptest_new <- ptest_final[, selected_features]
predictionsptest <- predict(ensemble_model, newdata = ptest_new)

# Combine the PrimeTest dataset with the predicted values
ptestres <- cbind(ptest, predictionsptest)
view(ptestres)
