library(data.table)
library(caret)
library(rpart)
library(rpart.plot)
library(ROSE)  
library(pROC)  

data <- read.csv("C:\\Users\\lenovo\\Downloads\\prodigy\\task 3\\bank-full.csv", sep=";", stringsAsFactors = FALSE)

data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$month <- as.factor(data$month)
data$poutcome <- as.factor(data$poutcome)
data$y <- as.factor(data$y)

sum(is.na(data))


data$edu_job <- interaction(data$education, data$job)

table(data$y)
data_balanced <- ROSE(y ~ ., data = data, seed = 123)$data

set.seed(123)
trainIndex <- createDataPartition(data_balanced$y, p = 0.7, list = FALSE)
trainData <- data_balanced[trainIndex, ]
testData <- data_balanced[-trainIndex, ]

# Model training with cross-validation
train_control <- trainControl(method = "cv", number = 10)
tuned_model <- train(y ~ ., data = trainData, method = "rpart", 
                     trControl = train_control, 
                     tuneLength = 10)

# Display the best model parameters
print(tuned_model$bestTune)

# Plot the decision tree using the final model
rpart.plot(tuned_model$finalModel)

# Make predictions on the test set
predictions <- predict(tuned_model, testData)

# Evaluate the model performance
conf_matrix <- confusionMatrix(predictions, testData$y)
print(conf_matrix)

# Calculate and plot the ROC curve and AUC
roc_curve <- roc(testData$y, as.numeric(predictions))
plot(roc_curve)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))
