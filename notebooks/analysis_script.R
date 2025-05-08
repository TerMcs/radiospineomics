library(mixOmics)
library(e1071)
library(caret)
library(ggrepel)
library(nnet)
library(corrplot)
library(colorblindr)
library(ggplot2)
library(ggforce)
library(rlang)
library(dplyr)
library(see)
library(ggnewscale)
library(pROC)
library(boot)
library(DescTools)
library(dplyr)
library(tidyr)
library(ggplot2)
library(yardstick)


feature_set <- "radiomics_dhi_dpsi" 

#### development set data ####
all_features <- "data/dev_features_IVD_area_adjusted.csv" 
df <- read.csv(all_features)

column_selection <- read.csv(feature_list)
dev_cols <- colnames(column_selection)
df <- df[, intersect(dev_cols, names(df))]

if (feature_set == "radiomics_dhi_dpsi") {
  standard_features <- "data/radiomics.csv"
  standard_features <- read.csv(standard_features)
  standard_features <- standard_features[, c("project_ID", "level", "ivd_height_index", "normalised_delta_si")]
  df <- merge(df, standard_features, by = c("project_ID", "level"))
} 

y <- as.factor(df$pfirrmann)
X <- as.matrix(df[, -c(1:9)])

#### test data ####
all_features_test <- "data/test_features_IVD_area_adjusted.csv" 

test_df <- test_df[, intersect(dev_cols, names(test_df))]

if (feature_set == "radiomics_dhi_dpsi") {
  test_df <- merge(test_df, standard_features, by = c("project_ID", "level"))
} 

y_test <- as.factor(test_df$pfirrmann)
X_test <- as.matrix(test_df)
X_test <- X_test[, colnames(X)]

#### quick initial look at the PCA and PLS-DA ####
pca_pfirrmann = pca(X, ncomp = 20, center = TRUE, scale = TRUE) 
plot(pca_pfirrmann) 

# standard PCA
plotIndiv(pca_pfirrmann, group = y, ind.names = FALSE,
          ellipse = TRUE, # plot the samples projected
          legend = TRUE, title = 'PCA on IVD radiomics, comp 1 - 2')

# sPLS-DA PCA
splsda_priffmann <- mixOmics::splsda(X, y, ncomp = 20) 
plotIndiv(splsda_priffmann, comp = 1:2, 
          group = y, ind.names = FALSE,  # colour points by class
          ellipse = TRUE, # include 95% confidence ellipse for each class
          legend = TRUE, title = 'PLSDA with confidence ellipses')

#### tune the optimal keep ####
# grid of possible keepX values that will be tested for each component
seq1 <- seq(1, 25, 1)
seq2 <- seq(10, 100, 5)
seq3 <- seq(100, 210, 10)
list.keepX <- c(seq1, seq2[-1], seq3[-1])

tune_splsda_pfirrmann <- tune.splsda(X, y, ncomp = 25, 
                                 validation = 'Mfold',
                                 folds = 5, nrepeat = 25, 
                                 dist = 'max.dist', 
                                 measure = "BER", 
                                 test.keepX = list.keepX,
                                 cpus = 1,
                                 progressBar = TRUE
                                 ) 

plot(tune_splsda_pfirrmann, col = color.jet(25)) 

optimal.ncomp <- tune_splsda_pfirrmann$choice.ncomp$ncomp
optimal.keepX <- tune_splsda_pfirrmann$choice.keepX[1:optimal.ncomp]

#### train the final model ####
final_splsda_pfirrmann <- mixOmics::splsda(X, y, 
                                    ncomp = optimal.ncomp, 
                                    keepX = optimal.keepX)

filename <- paste("data/MODEL_", feature_set, ".rds", sep="")
saveRDS(final_splsda_pfirrmann, file = filename)

#### Load the trained model if necessary ####
filename <- paste("data/MODEL_", feature_set, ".rds", sep="")
final_splsda_pfirrmann <- readRDS(filename)

#### evaluate the trained model ####
perf_splsda_pfirrmann <- perf(final_splsda_pfirrmann, 
                          folds = 5, nrepeat = 25, # use repeated cross-validation
                          validation = "Mfold", dist = "max.dist",  # use max.dist measure
                          progressBar = TRUE)

plot(perf_splsda_pfirrmann$features$stable[[1]], type = 'h', 
     ylab = 'Stability', 
     xlab = 'Features', 
     main = '(a) Comp 1', las =2)
plot(perf_splsda_pfirrmann$features$stable[[2]], type = 'h', 
     ylab = 'Stability', 
     xlab = 'Features', 
     main = '(b) Comp 2', las =2)
plot(perf_splsda_pfirrmann$features$stable[[3]], type = 'h', 
     ylab = 'Stability', 
     xlab = 'Features',
     main = '(c) Comp 3', las =2)

#### feature stability ####
stability_list <- list()

for (comp in 1:3) {
  stability_values <- perf_splsda_pfirrmann$features$stable[[comp]]
  print(paste("Component:", comp, "Length of stability values:", length(stability_values)))
  print(stability_values) 
}

for (comp in 1:length(perf_splsda_pfirrmann$features$stable)) {
  stability_values <- perf_splsda_pfirrmann$features$stable[[comp]]
  
  if (length(stability_values) == 0) next 
  
  features <- names(stability_values)
  print(features)
  stability_df <- data.frame(
    Feature = features,
    Stability = stability_values,
    Component = rep(comp, length(features)) 
  )
  
  if (comp > 1) {
    stability_df <- stability_df[, -2]
    colnames(stability_df) <- c("Feature", "Stability", "Component")
  }

  stability_list[[comp]] <- stability_df
}

final_stability_data <- do.call(rbind, stability_list)

stability_filename <- paste("data/EVALUATION_", features, "_stability.csv", sep = "")

write.csv(final_stability_data, file = stability_filename, row.names = FALSE)

#### model inference ####
predict_splsda_pfirrmann <- predict(final_splsda_pfirrmann, X_test)

predictions <- predict_splsda_pfirrmann$class$max.dist[,optimal.ncomp] 
probs <- predict_splsda_pfirrmann$predict[, , optimal.ncomp]
probs_df <- as.data.frame(probs)
colnames(probs_df) <- paste0("prob_", colnames(probs_df))
probs_eval_data <- cbind(ground_truth = y_test, probs_df)

#### confusion matrix ####
cm_sPLSDA <- caret::confusionMatrix(factor(predictions, levels = levels(y_test)), y_test)
print(cm_sPLSDA)
sensitivity_1 <- cm_sPLSDA$byClass[1, "Sensitivity"]
sensitivity_2 <- cm_sPLSDA$byClass[2, "Sensitivity"]
sensitivity_3 <- cm_sPLSDA$byClass[3, "Sensitivity"]
sensitivity_4 <- cm_sPLSDA$byClass[4, "Sensitivity"]

ba <- (sensitivity_1 + sensitivity_2 + sensitivity_3 + sensitivity_4) / 4


#### Conventional indices and simplified models ####
conventional_indices <- df[, c("pfirrmann", "IVD_height_index", "Normalised_delta_SI")]
conventional_indices$pfirrmann <- as.factor(conventional_indices$pfirrmann)
scaled_columns <- scale(conventional_indices[, c("IVD_height_index", "Normalised_delta_SI")], center = TRUE, scale = TRUE)
conventional_indices[, c("IVD_height_index", "Normalised_delta_SI")] <- scaled_columns

tune_grid <- expand.grid(
  C = 2^(-5:5),       
  sigma = 2^(-5:5) 
)

control <- trainControl(
  method = "repeatedcv",  
  number = 5,            
  repeats = 3,           
  verboseIter = TRUE    
)

set.seed(123)  
svm_model <- train(
  pfirrmann ~ ., 
  data = conventional_indices, 
  method = "svmRadial", 
  trControl = control, 
  tuneGrid = tune_grid
)

svm_conventional_indices <- svm(
  pfirrmann ~ ., 
  data = conventional_indices, 
  kernel = "radial", 
  cost = svm_model$bestTune$C, 
  gamma = svm_model$bestTune$sigma,
  probability = TRUE
)


#### test set inference ####
conventional_indices_test <- test_df[, c("pfirrmann", "IVD_height_index", "Normalised_delta_SI")]
conventional_indices_test$pfirrmann <- as.factor(conventional_indices_test$pfirrmann)

X_test_scaled <- scale(conventional_indices_test[, c("IVD_height_index", "Normalised_delta_SI")], center = TRUE, scale = TRUE)
svm_conventional_indices_pred <- predict(svm_conventional_indices, X_test_scaled)

svm_conventional_indices_prob <- predict(svm_conventional_indices, X_test_scaled, probability = TRUE)
probs <- attr(svm_conventional_indices_prob, "prob")
probs_eval_data <- as.data.frame(probs)
colnames(probs_eval_data) <- paste0("prob_", colnames(probs_eval_data))
probs_eval_data <- cbind(ground_truth = y_test, probs_eval_data)

cm_SVM <- caret::confusionMatrix(factor(svm_conventional_indices_pred, levels = levels(y)), y_test)

sensitivity_1 <- cm_SVM$byClass[1, "Sensitivity"]
sensitivity_2 <- cm_SVM$byClass[2, "Sensitivity"]
sensitivity_3 <- cm_SVM$byClass[3, "Sensitivity"]
sensitivity_4 <- cm_SVM$byClass[4, "Sensitivity"]

ba <- (sensitivity_1 + sensitivity_2 + sensitivity_3 + sensitivity_4) / 4


#### Top radiomics and simplified models ####
top_radiomics <- df[, c("pfirrmann", "Original_Sphericity", "Original_FirstOrder_InterquartileRange")]
top_radiomics$pfirrmann <- as.factor(top_radiomics$pfirrmann)
scaled_columns <- scale(top_radiomics[, c("Original_Sphericity", "Original_FirstOrder_InterquartileRange")], center = TRUE, scale = TRUE)
top_radiomics[, c("Original_Sphericity", "Original_FirstOrder_InterquartileRange")] <- scaled_columns

tune_grid <- expand.grid(
  C = 2^(-5:5),       
  sigma = 2^(-5:5)    
)

control <- trainControl(
  method = "repeatedcv",  
  number = 5,            
  repeats = 3,           
  verboseIter = TRUE     
)

set.seed(123)  
svm_model <- train(
  pfirrmann ~ ., 
  data = top_radiomics, 
  method = "svmRadial", 
  trControl = control, 
  tuneGrid = tune_grid
)

print(svm_model$bestTune)
print(svm_model)

svm_top_radiomics <- svm(
  pfirrmann ~ ., 
  data = top_radiomics, 
  kernel = "radial", 
  cost = svm_model$bestTune$C, 
  gamma = svm_model$bestTune$sigma,
  probability = TRUE
)

#### test set inference ####
top_radiomics <- test_df[, c("pfirrmann","Original_Sphericity", "Original_FirstOrder_InterquartileRange")]
top_radiomics$pfirrmann <- as.factor(top_radiomics$pfirrmann)
scaled_columns <- scale(top_radiomics[,  c("Original_Sphericity", "Original_FirstOrder_InterquartileRange")],
 center = TRUE, scale = TRUE)
top_radiomics[, c("Original_Sphericity", "Original_FirstOrder_InterquartileRange")] <- scaled_columns

svm_top_radiomics_pred <- predict(svm_top_radiomics, top_radiomics[, -1])

svm_top_radiomics_prob <- predict(svm_top_radiomics, top_radiomics[, -1], probability = TRUE)
probs <- attr(svm_top_radiomics_prob, "prob")
probs_eval_data <- as.data.frame(probs)
colnames(probs_eval_data) <- paste0("prob_", colnames(probs_eval_data))
probs_eval_data <- cbind(ground_truth = y_test, probs_eval_data)

cm_SVM <- caret::confusionMatrix(factor(svm_top_radiomics_pred, levels = levels(y)), y_test)

sensitivity_1 <- cm_SVM$byClass[1, "Sensitivity"]
sensitivity_2 <- cm_SVM$byClass[2, "Sensitivity"]
sensitivity_3 <- cm_SVM$byClass[3, "Sensitivity"]
sensitivity_4 <- cm_SVM$byClass[4, "Sensitivity"]

ba <- (sensitivity_1 + sensitivity_2 + sensitivity_3 + sensitivity_4) / 4



######################### metrics with confidence intervals #########################################
columns <- c("Conventional_indices", "top_2_radiomics", "texture_only_model", "full_radiomics_model")
rows <- c("Accuracy", "Macro_f1", "Micro_f1", "Macro_AUC", "balanced_accuracy", "kappa", "weighted_kappa", "lins_CCC",
          "sensitivity_1", "sensitivity_2", "sensitivity_3", "sensitivity_4",
          "specificity_1", "specificity_2", "specificity_3", "specificity_4")

df_results <- data.frame(matrix(NA, nrow = length(rows), ncol = length(columns)))
rownames(df_results) <- rows
colnames(df_results) <- columns
#####################################################################################################

features_being_evaluated <- "full_radiomics_model"
# features_being_evaluated <- "Conventional_indices"
# features_being_evaluated <- "top_2_radiomics"
# features_being_evaluated <- "texture_only_model"

accuracy <- function(data, indices) {
  sample_data <- data[indices, ]
  correct <- sum(sample_data$ground_truth == sample_data$predictions)
  total <- nrow(sample_data)
  return(correct / total)
}

balanced_accuracy <- function(data, indices) {
  sample_data <- data[indices, ]
  cm <- table(sample_data$ground_truth, sample_data$predictions)
  sensitivity_per_class <- diag(cm) / rowSums(cm)
  balanced_acc <- mean(sensitivity_per_class, na.rm = TRUE)
  return(balanced_acc)
}

macro_f1 <- function(data, indices) {
  sample_data <- data[indices, ]
  
  sample_data$ground_truth <- factor(sample_data$ground_truth)
  sample_data$predictions <- factor(sample_data$predictions, levels = levels(sample_data$ground_truth))
  
  cm <- table(sample_data$ground_truth, sample_data$predictions)
  classes <- levels(sample_data$ground_truth)
  f1 <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    class <- classes[i]
    tp <- cm[class, class]
    fp <- sum(cm[, class]) - tp
    fn <- sum(cm[class, ]) - tp
    
    prec <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1[i] <- if ((prec + rec) == 0) 0 else 2 * prec * rec / (prec + rec)
  }
  
  return(mean(f1))
}

micro_f1 <- function(data, indices) {
  sample_data <- data[indices, ]
  
  sample_data$ground_truth <- factor(sample_data$ground_truth)
  sample_data$predictions <- factor(sample_data$predictions, levels = levels(sample_data$ground_truth))
  
  cm <- table(sample_data$ground_truth, sample_data$predictions)
  
  tp_micro <- sum(diag(cm))
  fp_micro <- sum(colSums(cm)) - tp_micro
  fn_micro <- sum(rowSums(cm)) - tp_micro
  
  prec_micro <- if ((tp_micro + fp_micro) == 0) 0 else tp_micro / (tp_micro + fp_micro)
  rec_micro  <- if ((tp_micro + fn_micro) == 0) 0 else tp_micro / (tp_micro + fn_micro)
  
  return(if ((prec_micro + rec_micro) == 0) 0 else 2 * prec_micro * rec_micro / (prec_micro + rec_micro))
}

cohens_kappa <- function(data, indices) {
  sample_data <- data[indices, ]
  cm <- table(sample_data$ground_truth, sample_data$predictions)
  total <- sum(cm)
  observed_agreement <- sum(diag(cm)) / total
  row_marginals <- rowSums(cm) / total
  col_marginals <- colSums(cm) / total
  expected_agreement <- sum(row_marginals * col_marginals)
  kappa <- (observed_agreement - expected_agreement) / (1 - expected_agreement)
  return(kappa)
}

weighted_kappa <- function(data, indices, weights = "linear") {
  sample_data <- data[indices, ]
  cm <- table(sample_data$ground_truth, sample_data$predictions)
  total <- sum(cm)
  observed_proportions <- cm / total
  row_marginals <- rowSums(observed_proportions)
  col_marginals <- colSums(observed_proportions)
  expected_proportions <- outer(row_marginals, col_marginals, "*")
  n <- nrow(cm)
  weight_matrix <- switch(weights,
                          "quadratic" = outer(1:n, 1:n, function(i, j) (i - j)^2 / (n - 1)^2),
                          "linear" = outer(1:n, 1:n, function(i, j) abs(i - j) / (n - 1)),
                          stop("Invalid weights specified: use 'quadratic' or 'linear'")
  )
  observed_weighted_agreement <- sum(weight_matrix * observed_proportions)
  expected_weighted_agreement <- sum(weight_matrix * expected_proportions)
  kappa <- 1 - (observed_weighted_agreement / expected_weighted_agreement)
  return(kappa)
}

metrics <- list(
  Accuracy = accuracy,
  Macro_f1 = macro_f1,
  Micro_f1 = micro_f1,
  balanced_accuracy = balanced_accuracy,
  kappa = cohens_kappa,
  weighted_kappa = weighted_kappa
)

calculate_and_assign_metric <- function(data, metric_function, metric_name, column_name, results_df, R = 1000, seed = 123) {
  set.seed(seed)
  boot_results <- boot(data = data, statistic = metric_function, R = R)
  boot_mean <- mean(boot_results$t)

  ci <- boot.ci(boot_results, type = "perc")$percent[4:5]  # 95% CI
  formatted_mean <- sprintf("%.2f", boot_mean * 100)
  formatted_ci_lower <- sprintf("%.2f", ci[1] * 100)
  formatted_ci_upper <- sprintf("%.2f", ci[2] * 100)
  result_with_ci <- paste0(formatted_mean, " (", formatted_ci_lower, ", ", formatted_ci_upper, ")")
  results_df[metric_name, column_name] <- result_with_ci
  return(results_df)
}

for (metric_name in names(metrics)) {
  df_results <- calculate_and_assign_metric(
    data = predictions_gt, 
    metric_function = metrics[[metric_name]], 
    metric_name = metric_name, 
    column_name = features_being_evaluated, 
    results_df = df_results
  )
}

#### Lin's CCC ####
ccc_result <- CCC(as.numeric(as.factor(predictions_gt$predictions)), as.numeric(predictions_gt$ground_truth))
ccc_result_with_ci <- paste0(sprintf("%.2f", ccc_result$rho.c$est), 
  " (", 
  sprintf("%.2f",ccc_result$rho.c$lwr.ci), 
  ", ", 
  sprintf("%.2f",ccc_result$rho.c$upr.ci), 
  ")")
df_results["lins_CCC", features_being_evaluated] <- ccc_result_with_ci

#### Macro-AUC ####
macro_auc <- function(data, indices) {
  sample_data <- data[indices, ]
  sample_data$ground_truth <- factor(sample_data$ground_truth)
  
  class_levels <- levels(sample_data$ground_truth)
  prob_cols <- grep("^prob_", names(sample_data), value = TRUE)
  
  aucs <- numeric(length(class_levels))
  
  for (i in seq_along(class_levels)) {
    class_label <- class_levels[i]
    response_bin <- sample_data$ground_truth == class_label
    probs_class <- sample_data[[paste0("prob_", class_label)]]
    roc_obj <- roc(response = response_bin, predictor = probs_class)
    aucs[i] <- auc(roc_obj)
  }
  mean(aucs)
}

boot_macro_auc <- boot(data = probs_eval_data, statistic = macro_auc, R = 1000)
ci_result <- boot.ci(boot_macro_auc, type = "perc")
lower_bound <- ci_result$percent[4]
upper_bound <- ci_result$percent[5]
auc_with_ci <- paste0(
  sprintf("%.2f",boot_macro_auc$t0), 
  " (", 
  sprintf("%.2f",lower_bound), 
  ", ", 
  sprintf("%.2f",upper_bound), 
  ")")
df_results["Macro_AUC", features_being_evaluated] <- auc_with_ci[[1]]


print(paste0("Macro-average AUC: ", round(macro_auc, 3)))

#### Sensitivity and specificity ####
calculate_sensitivity <- function(data, indices, class_label) {
  sample_data <- data[indices, ]
  cm <- table(sample_data$ground_truth, sample_data$predictions)
  
  TP <- cm[class_label, class_label]
  FN <- sum(cm[class_label, ]) - TP
  sensitivity <- TP / (TP + FN)
  
  return(sensitivity)
}

calculate_specificity <- function(data, indices, class_label) {
  sample_data <- data[indices, ]
  cm <- table(sample_data$ground_truth, sample_data$predictions)
  
  TN <- sum(cm[-class_label, -class_label])
  FP <- sum(cm[class_label, -class_label])
  specificity <- TN / (TN + FP)
  
  return(specificity)
}

# Function to calculate, format, and assign bootstrapped results for sensitivity and specificity
calculate_metrics <- function(data, class_label, metric_name, column_name, results_df, R = 1000, seed = 123) {
  set.seed(seed)

  if (metric_name == "sensitivity") {
    metric_function <- calculate_sensitivity
  } else if (metric_name == "specificity") {
    metric_function <- calculate_specificity
  }

  boot_results <- boot(data = data, statistic = function(data, indices) metric_function(data, indices, class_label), R = R)
  boot_metric <- boot_results$t
  boot_mean <- mean(boot_metric)
  ci <- boot.ci(boot_results, type = "perc")$percent[4:5] 

  formatted_mean <- sprintf("%.2f", boot_mean*100)
  formatted_ci_lower <- sprintf("%.2f", ci[1]*100)
  formatted_ci_upper <- sprintf("%.2f", ci[2]*100)

  result_with_ci <- paste0(formatted_mean, " (", formatted_ci_lower, ", ", formatted_ci_upper, ")")
  results_df[metric_name, column_name] <- result_with_ci
  
  return(results_df)
}

# Loop through the classes and calculate sensitivity and specificity for each class (1 to 4)
for (class_label in 1:4) {
  # Sensitivity for each class
  metric_name <- paste0("sensitivity_", class_label)
  df_results <- calculate_metrics(
    data = predictions_gt, 
    class_label = class_label, 
    metric_name = metric_name, 
    column_name = features_being_evaluated, 
    results_df = df_results
  )
  
  # Specificity for each class
  metric_name <- paste0("specificity_", class_label)
  df_results <- calculate_and_assign_sensitivity_specificity(
    data = predictions_gt, 
    class_label = class_label, 
    metric_name = metric_name, 
    column_name = features_being_evaluated, 
    results_df = df_results
  )
}

# save the results to file
# write.csv(df_results, "data/RESULTS.csv")


#### Plots ####
#### variable importance ####
# form new perf() object which utilises the final model
perf.splsda.srbct <- perf(final_splsda_pfirrmann, 
                          folds = 5, nrepeat = 10, 
                          validation = "Mfold", dist = "max.dist",  
                          progressBar = TRUE)

# plot the stability of each feature for the first three components, 'h' type refers to histogram
par(mfrow=c(1,3))
plot(perf.splsda.srbct$features$stable[[1]], type = 'h', 
     ylab = 'Stability', 
     xlab = 'Features', 
     main = '(a) Comp 1', las =2)
plot(perf.splsda.srbct$features$stable[[2]], type = 'h', 
     ylab = 'Stability', 
     xlab = 'Features', 
     main = '(b) Comp 2', las =2)
plot(perf.splsda.srbct$features$stable[[3]], type = 'h', 
     ylab = 'Stability', 
     xlab = 'Features',
     main = '(c) Comp 3', las =2)


#### sPLSDA components 1 and 2 plot ####
data_points <- final_splsda_pfirrmann$variates$X[, 1:2]

###############
# ##### to plot only the test data into the trained latent space:
# data_points <- predict_splsda_pfirrmann$variates[, 1:2]
# # add a column for the group based on the test data
# y <- as.factor(y_test)
###############

data_points <- cbind(data_points, group = y)
data_points <- as.data.frame(data_points)
data_points$group <- as.factor(data_points$group) 

group_sizes <- data_points %>% 
  count(group) %>% 
  arrange(desc(n))

data_points <- data_points %>% 
  mutate(group = factor(group, levels = group_sizes$group)) %>% 
  arrange(group)

p <- ggplot(data_points, aes(x = comp1, y = comp2, color = group, shape = group)) +
  geom_point(size = 3, alpha = 0.7, stroke = 1.5, fill = NA) + 
  scale_shape_manual(values = c(3, 4, 21, 24), name = "Pfirrmann \ngrade", labels = c("2", "3", "4", "5")) + 
  scale_color_brewer(palette = "Set1", name = "Pfirrmann \ngrade", labels = c("2", "3", "4", "5")) +   
  stat_ellipse(aes(color = group), type = "t", level = 0.95, size = 1.5, alpha = 1, show.legend = FALSE, ) + 
  labs(
    x = 'Component 1: 21% of explained variance',
    y = 'Component 2: 22% of explained variance'
  ) +
  theme_minimal() +

  theme(
    legend.position = c(0.12,0.85),
    axis.title = element_text(size = 20),  
    axis.text = element_text(size = 20),   
    legend.title = element_text(size = 20),  
    legend.text = element_text(size = 20),  
  )  +
  guides(
    color = guide_legend(override.aes = list(alpha = 1))
  )
print(p)
cvd_grid(p)

#### conventional indices plot ####
df$pfirrmann <- as.factor(df$pfirrmann)
ggplot(df, aes(x = Normalised_delta_SI, y = IVD_height_index, color = pfirrmann, shape = pfirrmann)) +
  geom_point(size = 2, alpha = 0.3, stroke = 1.2, fill = NA) + 
  scale_shape_manual(values = c(3, 4, 21, 24), name = "Pfirrmann \ngrade", labels = c("2", "3", "4", "5")) + 
  stat_ellipse(type = "t", level = 0.95, size = 1, show.legend = FALSE) + 
  scale_color_OkabeIto(name = "Pfirrmann \ngrade", labels = c("2", "3", "4", "5"),darken = 0, order = c(2,1,3,6)) +   
  scale_y_reverse() +
  scale_x_reverse() +
  labs(
    x = 'Peak Signal Intensity Difference',
    y = 'Disc Height Index'
  ) +
  theme_minimal() +

  theme(
    legend.position = c(0.12,0.85),
    axis.title = element_text(size = 16), 
    axis.text = element_text(size = 16),   
    legend.title = element_text(size = 16),  
    legend.text = element_text(size = 16),   
  )  +
  guides(
    color = guide_legend(override.aes = list(alpha = 1))
  )


#### top radiomics plot ####
df$pfirrmann <- as.factor(df$pfirrmann)
ggplot(df, aes(x = Original_FirstOrder_InterquartileRange, y = Original_Sphericity, color = pfirrmann, shape = pfirrmann)) +
  geom_point(size = 2, alpha = 0.3, stroke = 1.2, fill = NA) + 
  scale_shape_manual(values = c(3, 4, 21, 24), name = "Pfirrmann \ngrade", labels = c("2", "3", "4", "5")) + 
  stat_ellipse(type = "t", level = 0.95, size = 1, show.legend = FALSE) + 
  scale_color_OkabeIto(name = "Pfirrmann \ngrade", labels = c("2", "3", "4", "5"),darken = 0, order = c(2,1,3,6)) +   
  scale_y_reverse() +
  scale_x_reverse() +
  labs(
    y = 'Sphericity',
    x = 'Original First-Order Interquartile Range'
  ) +
  theme_minimal() +

  theme(
    legend.position = c(0.12,0.85),
    axis.title = element_text(size = 16),  
    axis.text = element_text(size = 16),   
    legend.title = element_text(size = 16),  
    legend.text = element_text(size = 16),   
  )  +
  guides(
    color = guide_legend(override.aes = list(alpha = 1))
  )


########################### ROC curve plot #########################################

probs_eval_data <- probs_eval_data %>% 
  mutate(ground_truth = factor(ground_truth))

auc_results <- list()

for (class_number in 2:5) {
  probs_column <- paste0("prob_", class_number)
  roc_data <- roc(response = probs_eval_data$ground_truth == class_number, 
                  predictor = probs_eval_data[[probs_column]])
  auc_ci <- ci.auc(roc_data, boot.n = 1000)  
  auc_results[[as.character(class_number)]] <- auc_ci
}

auc_df <- do.call(rbind, lapply(names(auc_results), function(class_number) {
  auc_ci <- auc_results[[class_number]]
  data.frame(class = as.factor(class_number), 
             AUC = auc_ci[2], 
             lower = auc_ci[1], 
             upper = auc_ci[3])
}))

roc_data_list <- list()

for (class_number in 2:5) {
  probs_column <- paste0("prob_", class_number)
  roc_data <- roc(response = probs_eval_data$ground_truth == class_number, 
                  predictor = probs_eval_data[[probs_column]])
  roc_df <- data.frame(sensitivity = roc_data$sensitivities, 
                       specificity = roc_data$specificities,
                       class = as.factor(class_number))
  roc_data_list[[length(roc_data_list) + 1]] <- roc_df
}

roc_combined_data <- do.call(rbind, roc_data_list)

positions <- data.frame(
  class = factor(2:5),
  x = c(0.7, 0.7, 0.7, 0.7),     
  y = c(0.20, 0.15, 0.10, 0.05)  
)


auc_df <- merge(auc_df, positions, by = "class")

ggplot(roc_combined_data, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line() +
  geom_abline(linetype = "dashed", color = "gray") +
  geom_text(data = auc_df, aes(x = x, y = y, 
                               label = paste0("AUC PG ", class, " = ", sprintf("%.1f", AUC * 100),   
                                              " (", sprintf("%.1f", lower * 100), "-", sprintf("%.1f", upper * 100), ")")
                               ), 
            color = "black", 
            show.legend = FALSE) +
  labs(title = "ROC curves per Pfirrmann grade, best two radiomics",
       x = "1 - Specificity",
       y = "Sensitivity",
       color = "Pfirrmann\ngrade") +
  theme_minimal()