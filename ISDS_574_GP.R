#---------------------------
#prework
#---------------------------

#install all packages needed (only run if you don't have these)
#install.packages(c("dplyr","ggplot2","caret","randomForest","rpart","pROC","ROCR","xgboost","DataExplorer"))

#load libraries
library(dplyr); library(ggplot2); library(caret); library(randomForest)
library(rpart); library(pROC); library(ROCR); library(xgboost); library(DataExplorer)

#read + perform initial inspections
loan_df <- as.data.frame(read.csv('C:/Temp/Loan.csv', stringsAsFactors = FALSE))
dim(loan_df)
str(loan_df)
summary(loan_df)

#----------------------------------------
#ensure data type and factor conversions
#----------------------------------------
loan_df$ApplicationDate<-as.Date(loan_df$ApplicationDate)
factor_cols<-c("EmploymentStatus", "EducationLevel", "MaritalStatus","HomeOwnershipStatus",
               "BankruptcyHistory", "LoanPurpose","PaymentHistory","LoanApproved")
loan_df[factor_cols]<-lapply(loan_df[factor_cols], function(x) as.factor(x))

#missing value check
sapply(loan_df, function(x) sum(is.na(x)))
DataExplorer::plot_missing(loan_df)

#---------------------------------------
#outlier detection + transformations
#---------------------------------------
#outlier detection & potential transformations
#select numeric columns
numeric_cols <- sapply(loan_df, is.numeric)
num_df <- loan_df[, numeric_cols]

#function to plot boxplots for all numeric columns
library(ggplot2)

plot_boxplot <- function(df){
  for(col in names(df)){
    p <- ggplot(df, aes_string(x = "''", y = col)) +
      geom_boxplot(outlier.colour = "red", outlier.shape = 16) +
      ylab(col) +
      ggtitle(paste("Boxplot of", col))
    print(p)
  }
}

#run the boxplots
plot_boxplot(num_df)

#identify skewed variables
library(e1071)
skew_vals <- sapply(num_df, skewness, na.rm = TRUE)
skew_vals[abs(skew_vals) > 1]  # |skew|>1 considered highly skewed

#apply log transformation to skewed variables
skewed_cols <- names(skew_vals[abs(skew_vals) > 1])

for(col in skewed_cols){
  new_col <- paste0(col, "_log")
  loan_df[[new_col]] <- log1p(loan_df[[col]])
  
  # plot transformed variable
  p <- ggplot(loan_df, aes_string(x = "''", y = new_col)) +
    geom_boxplot(outlier.colour = "blue", outlier.shape = 16) +
    ylab(new_col) +
    ggtitle(paste("Boxplot of", new_col, "(log-transformed)"))
  print(p)
}

#check log variables
log_cols <- paste0(skewed_cols, "_log")  # same as in Step 4

# Compute skewness for log-transformed columns
log_skew_vals <- sapply(loan_df[, log_cols], skewness, na.rm = TRUE)

# View results
log_skew_vals

#previousloandefaults & totaldebttoincomeratio are still highly skewed
#square-root transformation on totaldebttoincomeratio
# Square root transformation (after shifting to avoid negatives)
loan_df$TotalDebtToIncomeRatio_sqrt <- sqrt(loan_df$TotalDebtToIncomeRatio)
#recheck skewness
skewness(loan_df$TotalDebtToIncomeRatio_sqrt, na.rm = TRUE)
#leave previousloandefaults as is, its fine for log reg and trees, will 
#need to change to categorical if we do k-nn

#----------------------------------------
#dummy encoding for categorical variables
#----------------------------------------
# Select categorical variables to encode
cat_vars <- c("EmploymentStatus", "EducationLevel", "MaritalStatus", 
              "HomeOwnershipStatus", "LoanPurpose")

# Convert to factors
loan_df[cat_vars] <- lapply(loan_df[cat_vars], factor)

# Create dummy variables (one-hot encoding)
dummies <- model.matrix(~ EmploymentStatus + EducationLevel + MaritalStatus + 
                          HomeOwnershipStatus + LoanPurpose - 1, data = loan_df)

# Convert to data.frame and bind to original
dummies_df <- as.data.frame(dummies)
loan_df <- cbind(loan_df, dummies_df)

# Optional: remove original categorical variables
loan_df <- loan_df[ , !(names(loan_df) %in% cat_vars)]

#-----------------------
#build models
#-----------------------
#data model
set.seed(123)

#create train/test split
n <- nrow(loan_df)
train_indices <- sample(1:n, size = 0.7*n)
train_df <- loan_df[train_indices, ]
test_df <- loan_df[-train_indices, ]

#------------------------------------
#logistic regression on loan approved
#------------------------------------
names(dummies_df)
# Train logistic regression
logit_model <- glm(LoanApproved ~ AnnualIncome_log + LoanAmount_log + TotalDebtToIncomeRatio_sqrt + PreviousLoanDefaults +
                     EmploymentStatusEmployed + `EmploymentStatusSelf-Employed` + EmploymentStatusUnemployed +
                     EducationLevelBachelor + EducationLevelDoctorate + `EducationLevelHigh School` + EducationLevelMaster +
                     MaritalStatusMarried + MaritalStatusSingle + MaritalStatusWidowed +
                     HomeOwnershipStatusOther + HomeOwnershipStatusOwn + HomeOwnershipStatusRent +
                     `LoanPurposeDebt Consolidation` + LoanPurposeEducation + LoanPurposeHome + LoanPurposeOther,
                   data = train_df, family = binomial)

summary(logit_model)

# Predict probabilities on test set
pred_probs <- predict(logit_model, newdata = test_df, type = "response")

# Convert to class using 0.5 threshold
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

# Confusion matrix
table(Predicted = pred_class, Actual = test_df$LoanApproved)
cm_logit <- table(Predicted = pred_class, Actual = test_df$LoanApproved)

#-------------------------------------
#decision tree on loan approved
#-------------------------------------
library(rpart)

tree_model <- rpart(LoanApproved ~ AnnualIncome_log + LoanAmount_log + TotalDebtToIncomeRatio_sqrt + PreviousLoanDefaults +
                      EmploymentStatusEmployed + `EmploymentStatusSelf-Employed` + EmploymentStatusUnemployed +
                      EducationLevelBachelor + EducationLevelDoctorate + `EducationLevelHigh School` + EducationLevelMaster +
                      MaritalStatusMarried + MaritalStatusSingle + MaritalStatusWidowed +
                      HomeOwnershipStatusOther + HomeOwnershipStatusOwn + HomeOwnershipStatusRent +
                      `LoanPurposeDebt Consolidation` + LoanPurposeEducation + LoanPurposeHome + LoanPurposeOther,
                    data = train_df, method = "class")

# Predict class on test set
tree_pred <- predict(tree_model, newdata = test_df, type = "class")

# Confusion matrix
table(Predicted = tree_pred, Actual = test_df$LoanApproved)
cm_tree  <- table(Predicted = tree_pred, Actual = test_df$LoanApproved)

#----------------------------------
#random forest for loanapproved
#----------------------------------
library(randomForest)

predictors <- c("AnnualIncome_log", "LoanAmount_log", "TotalDebtToIncomeRatio_sqrt", "PreviousLoanDefaults",
                "EmploymentStatusEmployed", "EmploymentStatusSelf-Employed", "EmploymentStatusUnemployed",
                "EducationLevelBachelor", "EducationLevelDoctorate", "EducationLevelHigh School", "EducationLevelMaster",
                "MaritalStatusMarried", "MaritalStatusSingle", "MaritalStatusWidowed",
                "HomeOwnershipStatusOther", "HomeOwnershipStatusOwn", "HomeOwnershipStatusRent",
                "LoanPurposeDebt Consolidation", "LoanPurposeEducation", "LoanPurposeHome", "LoanPurposeOther")

rf_model <- randomForest(x = train_df[, predictors],
                         y = factor(train_df$LoanApproved),
                         ntree = 100)

# Predict class on test set
rf_pred <- predict(rf_model, newdata = test_df)

# Confusion matrix
table(Predicted = rf_pred, Actual = test_df$LoanApproved)
cm_rf    <- table(Predicted = rf_pred, Actual = test_df$LoanApproved)

#evaluate ROC/AUC
library(pROC)

roc_obj <- roc(test_df$LoanApproved, pred_probs)
auc(roc_obj)
plot(roc_obj, main = "Logistic Regression ROC Curve")


#interpretation
#model performance
# Function to calculate performance metrics from confusion matrix
calc_metrics <- function(cm) {
  # cm should be a 2x2 table: Predicted x Actual
  TN <- cm[1,1]
  FP <- cm[1,2]
  FN <- cm[2,1]
  TP <- cm[2,2]
  
  accuracy  <- (TP + TN) / sum(cm)
  sensitivity <- TP / (TP + FN)   # True positive rate
  specificity <- TN / (TN + FP)   # True negative rate
  precision <- TP / (TP + FP)
  
  return(data.frame(Accuracy = accuracy,
                    Sensitivity = sensitivity,
                    Specificity = specificity,
                    Precision = precision))
}

# Apply to your confusion matrices
metrics_logit <- calc_metrics(cm_logit)
metrics_tree  <- calc_metrics(cm_tree)
metrics_rf    <- calc_metrics(cm_rf)

# Combine for easy comparison
model_comparison <- rbind(Logistic = metrics_logit,
                          DecisionTree = metrics_tree,
                          RandomForest = metrics_rf)
model_comparison

#identify top predictors - log
summary(logit_model)$coefficients
coef_df <- as.data.frame(summary(logit_model)$coefficients)
coef_df$Variable <- rownames(coef_df)
coef_df <- coef_df[order(-abs(coef_df$Estimate)), ]
head(coef_df, 10)  # top 10 predictors

#identify important variables - random forest
library(randomForest)

# View importance
importance(rf_model)

# Plot top predictors
varImpPlot(rf_model)

library(pROC)

roc_obj <- roc(test_df$LoanApproved, pred_probs)  # pred_probs from logistic regression
auc(roc_obj)  # AUC value
plot(roc_obj, main = "Logistic Regression ROC Curve")

#check correlations
numeric_vars <- c("AnnualIncome_log", "LoanAmount_log", "CreditScore",
                  "TotalDebtToIncomeRatio_sqrt", "MonthlyDebtPayments_log")
cor(loan_df[, numeric_vars])



