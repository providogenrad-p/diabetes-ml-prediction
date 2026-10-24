# ============================================================
# Diabetes Prediction - Pima Indians Diabetes Dataset
# ============================================================

# 0. LIBRARIES
library(corrplot)      # correlation matrix
library(caret)         # stratified split, confusion matrices
library(rpart)         # decision tree
library(rpart.plot)    # decision tree visualization
library(randomForest)  # random forest
library(class)         # k-NN
library(e1071)         # naive Bayes
library(pROC)          # ROC curves and AUC

# 1. DATA LOADING AND INITIAL OVERVIEW
data <- read.csv("diabetes.csv")

cat("Dataset:", nrow(data), "rows,", ncol(data), "columns\n\n")
print(head(data))
print(summary(data))

# 2. Data Cleaning: mark physiologically impossible zeros as missing
# (Pregnancies = 0 is valid, so that column is not changed)
zero_cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

for (col in zero_cols) {
  cat(col, "- zeros replaced with NA:", sum(data[[col]] == 0), "\n")
  data[[col]][data[[col]] == 0] <- NA
}

# 3. Exploratory Data Analysis (EDA)
# EDA is done on the raw data; missing values are simply ignored here.

# Outcome distribution
barplot(table(data$Outcome),
        main = "Diabetes Outcome Distribution",
        col = c("pink", "purple"),
        names.arg = c("Healthy", "Diabetes"),
        xlab = "Outcome",
        ylab = "Count")

# Histograms for Glucose and BMI
par(mfrow = c(1, 2))
hist(data$Glucose, col = "pink",   main = "Glucose", xlab = "Glucose")
hist(data$BMI,     col = "purple", main = "BMI",     xlab = "BMI")
par(mfrow = c(1, 1))

# Glucose by Outcome
boxplot(Glucose ~ Outcome, data = data,
        col = c("pink", "purple"),
        main = "Glucose vs Outcome",
        xlab = "Outcome", ylab = "Glucose",
        names = c("Healthy", "Diabetes"),
        boxwex = 0.5)

# BMI by Outcome
boxplot(BMI ~ Outcome, data = data,
        col = c("pink", "purple"),
        main = "BMI vs Outcome",
        xlab = "Outcome", ylab = "BMI",
        names = c("Healthy", "Diabetes"),
        boxwex = 0.5)

# Correlation matrix (includes Outcome; pairwise complete observations because of NAs)
corrplot(cor(data, use = "pairwise.complete.obs"),
         method = "color", addCoef.col = "black",
         number.cex = 0.7, tl.col = "black")

# 4. Data Split (stratified 70/30)
set.seed(123)
train_index <- createDataPartition(factor(data$Outcome), p = 0.7, list = FALSE)
train <- data[train_index, ]
test  <- data[-train_index, ]

cat("\nClass distribution - training set:\n"); print(table(train$Outcome))
cat("\nClass distribution - test set:\n");     print(table(test$Outcome))

# 5. Missing Value Imputation (medians from the TRAINING set only)
# Using only training medians prevents information from the test set leaking into the models.
train_medians <- sapply(train[zero_cols], median, na.rm = TRUE)

for (col in zero_cols) {
  train[[col]][is.na(train[[col]])] <- train_medians[[col]]
  test[[col]][is.na(test[[col]])]   <- train_medians[[col]]
}

cat("\nMedians used for imputation:\n"); print(train_medians)
print(summary(train))

# Convert outcome to factor for classification models
train$Outcome <- factor(train$Outcome, levels = c(0, 1))
test$Outcome  <- factor(test$Outcome,  levels = c(0, 1))

predictors <- setdiff(names(train), "Outcome")

# 6. Logistic Regresion
model_log <- glm(Outcome ~ ., data = train, family = binomial)
print(summary(model_log))

# Odds ratios: how the odds of diabetes change per one-unit increase of each predictor
cat("\nOdds ratios:\n")
print(round(exp(coef(model_log)), 3))

prob_log <- predict(model_log, test, type = "response")
pred_log <- factor(ifelse(prob_log > 0.5, 1, 0), levels = c(0, 1))

# 7. Decision Tree
model_tree <- rpart(Outcome ~ ., data = train, method = "class")
rpart.plot(model_tree, main = "Decision Tree", extra = 104)

prob_tree <- predict(model_tree, test, type = "prob")[, "1"]
pred_tree <- predict(model_tree, test, type = "class")

# 8. Random Forest
set.seed(123)
model_rf <- randomForest(Outcome ~ ., data = train, ntree = 500, importance = TRUE)
print(model_rf)
varImpPlot(model_rf, main = "Random Forest - Variable Importance")

prob_rf <- predict(model_rf, test, type = "prob")[, "1"]
pred_rf <- predict(model_rf, test)

# 9. k-NN with scaling
# Scale predictors using training statistics only
train_scaled <- scale(train[, predictors])
test_scaled  <- scale(test[, predictors],
                      center = attr(train_scaled, "scaled:center"),
                      scale  = attr(train_scaled, "scaled:scale"))

# Tune k with leave-one-out cross-validation (odd values avoid ties)
k_values <- seq(3, 21, 2)
set.seed(123)
cv_acc <- sapply(k_values, function(k) {
  mean(knn.cv(train_scaled, train$Outcome, k = k) == train$Outcome)
})
best_k <- k_values[which.max(cv_acc)]

plot(k_values, cv_acc, type = "b", pch = 19, col = "purple",
     xlab = "k", ylab = "LOOCV Accuracy", main = "k-NN: Choosing k")
cat("\nOptimal k for k-NN:", best_k, "\n")

# Final k-NN prediction
set.seed(123)
pred_knn <- knn(train_scaled, test_scaled, train$Outcome, k = best_k, prob = TRUE)

# Convert the winning-class vote share into P(Outcome = 1) for ROC/AUC
win_prob <- attr(pred_knn, "prob")
prob_knn <- ifelse(pred_knn == "1", win_prob, 1 - win_prob)
attr(pred_knn, "prob") <- NULL

# 10. Naive Bayes
model_nb <- naiveBayes(Outcome ~ ., data = train)
prob_nb  <- predict(model_nb, test, type = "raw")[, "1"]
pred_nb  <- predict(model_nb, test)

# 11. ZeroR Baseline (always predicts the majority class of the training set)
majority_class <- names(which.max(table(train$Outcome)))
pred_zeror <- factor(rep(majority_class, nrow(test)), levels = c(0, 1))

# 12. Confusion Matrices And Metrics
models <- list(
  "Logistic Regression" = list(pred = pred_log,   prob = prob_log),
  "Decision Tree"       = list(pred = pred_tree,  prob = prob_tree),
  "Random Forest"       = list(pred = pred_rf,    prob = prob_rf),
  "k-NN"                = list(pred = pred_knn,   prob = prob_knn),
  "Naive Bayes"         = list(pred = pred_nb,    prob = prob_nb),
  "ZeroR Baseline"      = list(pred = pred_zeror, prob = NULL)
)

results  <- data.frame()
roc_list <- list()

for (name in names(models)) {
  cm <- confusionMatrix(models[[name]]$pred, test$Outcome,
                        positive = "1", mode = "everything")
  cat("\n========== Results:", name, "==========\n")
  print(cm)
  
  if (is.null(models[[name]]$prob)) {
    auc_value <- 0.5  # a constant predictor has no ranking ability (AUC = 0.5)
  } else {
    roc_list[[name]] <- roc(test$Outcome, models[[name]]$prob,
                            levels = c("0", "1"), direction = "<", quiet = TRUE)
    auc_value <- as.numeric(auc(roc_list[[name]]))
  }
  
  results <- rbind(results, data.frame(
    Model             = name,
    Accuracy          = unname(cm$overall["Accuracy"]),
    Balanced_Accuracy = unname(cm$byClass["Balanced Accuracy"]),
    Precision         = unname(cm$byClass["Precision"]),
    Recall            = unname(cm$byClass["Recall"]),
    Specificity       = unname(cm$byClass["Specificity"]),
    F1                = unname(cm$byClass["F1"]),
    AUC               = auc_value
  ))
}

# 13. ROC curves
roc_colors <- c("hotpink", "purple", "orange", "blue", "darkgreen")

plot(roc_list[[1]], col = roc_colors[1], lwd = 2,
     main = "ROC Curves", legacy.axes = TRUE)  # pROC draws the correct chance diagonal itself
for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = roc_colors[i], lwd = 2, add = TRUE)
}
legend("bottomright",
       legend = paste0(names(roc_list), " (AUC = ",
                       sprintf("%.3f", sapply(roc_list, auc)), ")"),
       col = roc_colors, lwd = 2, cex = 0.8)

# 14. Final comparison table (sorted by AUC)
results[, -1] <- round(results[, -1], 3)
results <- results[order(-results$AUC), ]

cat("\n========== MODEL COMPARISON ==========\n")
print(results, row.names = FALSE)

cat("\nBest model by AUC:", results$Model[1], "\n")
cat("Best model by Balanced Accuracy:",
    results$Model[which.max(results$Balanced_Accuracy)], "\n")

if (interactive()) View(results)