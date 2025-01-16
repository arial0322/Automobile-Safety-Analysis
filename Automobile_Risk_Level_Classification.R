library(ggplot2)
library(corrplot)
library(car)
require(psych)
library(MASS)
library(klaR)
library(tree)
library(rpart)				        
library(rpart.plot)	
library(randomForest)
library(stargazer)
library(gbm)

setwd("C:\\Users\\Arial\\Desktop\\McGill\\MGSC661\\Final Project")
automobile <- read.csv("Dataset 5 — Automobile data.csv")


############################### Project Overview ###############################

## Observe missing values of each column
missing_counts <- sapply(automobile, function(x) sum(x == '?'))
print(missing_counts)

## Observe the relationship between symboling and normalized.losses
df_temp <- automobile[automobile$normalized.losses != "?", ]
df_temp$normalized.losses <- as.numeric(df_temp$normalized.losses)
df_temp$symboling <- as.numeric(df_temp$symboling)
reg_temp <- lm(df_temp$normalized.losses ~ df_temp$symboling)
summary(reg_temp)

df_temp$symboling <- as.factor(df_temp$symboling)
ggplot(df_temp, aes(x=symboling, y=normalized.losses))+geom_boxplot(fill="lightblue")


################################ Pre-processing ################################

## Remove missing values from all columns except normalized.losses
automobile[-which(names(automobile) == "normalized.losses")] <- 
  lapply(automobile[-which(names(automobile) == "normalized.losses")], 
         function(x) ifelse(x == "?", NA, x))
automobile <- na.omit(automobile)

## number mapping
word_to_number <- c(one = 1, two = 2, three = 3, four = 4, five = 5, six = 6,
                    seven = 7, eight = 8, nine = 9, ten = 10, eleven = 11, twelve = 12)
automobile$num.of.cylinders <- word_to_number[automobile$num.of.cylinders]

automobile[, c(3,4,5,6,7,8,9,15,18)] <- lapply(automobile[, c(3,4,5,6,7,8,9,15,18)], as.factor)
categorical_col <- automobile[, c(3,4,5,6,7,8,9,15,18)]

automobile[, c(19,20,22,23,26)] <- lapply(automobile[, c(19,20,22,23,26)], as.numeric)
numerical_col <- automobile[, c(10,11,12,13,14,16,17,19,20,21,22,23,24,25,26)]

## Correlation
# correlation matrix
# pairs.panels(numerical_col)
corr_matrix=cor(numerical_col)
round(corr_matrix, 2)
corrplot(corr_matrix, 
         method = "color", 
         type = "upper", 
         addCoef.col = "black",
         number.cex = 0.7,
         tl.col = "black",
         tl.srt = 45)
# vif
automobile$symboling <- as.numeric(automobile$symboling)
mreg_formula <- as.formula(paste("symboling ~", paste(names(numerical_col), collapse="+")))
mreg_numerical <- lm(mreg_formula, data=automobile)
vif(mreg_numerical)

## Convert highly-correlated columns
automobile$base.area <- automobile$length * automobile$width
automobile$wheelbase.to.length <- automobile$wheel.base / automobile$length
automobile$engine.size.ratio <- automobile$engine.size / (automobile$curb.weight)
automobile$avg.mpg <- rowMeans(automobile[, c("city.mpg", "highway.mpg")])
# automobile <- automobile[, !names(automobile) %in% c("length", "width", "wheel.base", "engine.size", "curb.weight", "horsepower", "city.mpg", "highway.mpg")]

processed_numerical_col <- automobile[, c(13,16,19,20,21,23,26:30)]
categorical_col <- automobile[, c(3:9,15,18)]
automobile$symboling <- as.factor(automobile$symboling)
automobile$symboling_numeric <- as.numeric(as.character(automobile$symboling))
attach(automobile)

## Evaluate correlation again
corr_matrix=cor(processed_numerical_col)
round(corr_matrix, 2)
corrplot(corr_matrix, 
         method = "color", 
         type = "upper", 
         addCoef.col = "black",
         number.cex = 0.7,
         tl.col = "black",
         tl.srt = 45)
mreg_formula <- as.formula(paste("symboling_numeric ~", paste(names(processed_numerical_col), collapse="+")))
mreg_numerical <- lm(mreg_formula, data=automobile)
vif(mreg_numerical)


##################################### LDA ######################################

## Prior probabilities
table(symboling)
pi_table <- table(symboling) / sum(table(symboling))
pi_m2 <- pi_table[["-2"]]
pi_m1 <- pi_table[["-1"]]
pi_0 <- pi_table[["0"]]
pi_1 <- pi_table[["1"]]
pi_2 <- pi_table[["2"]]

## Plotting probability density functions
for (col in colnames(processed_numerical_col)) {
  print(
    ggplot(automobile, aes(x=!!sym(col))) +
      geom_histogram(bins = 50) +
      facet_grid(symboling) + 
      ggtitle(paste("Probability Density Functions of Fs(", paste(col, ")")))
  )
}

## LDA
scaled_numerical_col <- processed_numerical_col[-which(names(processed_numerical_col) == "engine.size.ratio")]
scaled_numerical_col <- scale(scaled_numerical_col)
combined_cols <- c(names(categorical_col), colnames(scaled_numerical_col))
formula_lda = as.formula(paste("symboling ~", paste(combined_cols, collapse="+")))
mylda <- lda(formula_lda)
mylda

## Visualize feature importance
coefficients <- as.matrix(mylda$scaling)
abs_coefficients <- abs(coefficients)
top_features <- order(rowSums(abs_coefficients), decreasing = TRUE)[1:10]
top_coefficients <- coefficients[top_features, ]
barplot(t(top_coefficients), beside = TRUE,
        col = c("lightblue", "lightgreen", "pink", "orange", "lightgrey"),
        main = "Top 10 Feature Impact on Linear Discriminants",
        ylab = "Coefficient Value",
        las = 2,
        cex.names = 0.7)
legend("center", legend = c("LD1", "LD2", "LD3", "LD4", "LD5"), 
       fill = c("lightblue", "lightgreen", "pink", "orange", "lightgrey"),
       title = "Linear Discriminants",
       cex = 0.65)

last_features <- order(rowSums(abs_coefficients), decreasing = TRUE)[41:50]
last_coefficients <- coefficients[last_features, ]
barplot(t(last_coefficients), beside = TRUE,
        col = c("lightblue", "lightgreen", "pink", "orange", "lightgrey"),
        main = "Last 10 Feature Impact on Linear Discriminants",
        ylab = "Coefficient Value",
        las = 2,
        cex.names = 0.7)
legend("topright", legend = c("LD1", "LD2", "LD3", "LD4", "LD5"), 
       fill = c("lightblue", "lightgreen", "pink", "orange", "lightgrey"),
       title = "Linear Discriminants",
       cex = 0.65)


################################ Classification ################################

# numerical_col <- automobile[, c(10,11,12,13,14,16,17,19:30)]
numerical_col <- automobile[, c(10,11,12,13,14,16,17,19:22,24,25,28:30)]
categorical_col <- automobile[, c(3:9,15,18)]

combined_cols <- c(names(categorical_col), names(numerical_col))
formula_tree <- as.formula(paste("symboling ~", paste(combined_cols, collapse="+")))

# #### Classification Tree #####
classifiedtree <- rpart(formula_tree, cp=0.001, na.action=na.omit)
rpart.plot(classifiedtree)
printcp(classifiedtree)
summary(classifiedtree)

## Prunning a tree
# Overfit the tree
myoverfittedtree <- rpart(formula_tree, data = automobile, control = rpart.control(cp = 1e-100))

# Display complexity parameter table and plot
printcp(myoverfittedtree)
plotcp(myoverfittedtree)

# Find optimal cp
opt_cp <- myoverfittedtree$cptable[which.min(myoverfittedtree$cptable[,"xerror"]), "CP"]
opt_cp

# Prune the tree
pruned_tree <- prune(myoverfittedtree, cp=opt_cp)

# Plot the pruned tree
rpart.plot(pruned_tree)
printcp(pruned_tree)
summary(pruned_tree)


#### random forest ####
classwt <- 1 / (pi_table * 6)
set.seed(1)
classifiedforest_1 <- randomForest(
  formula_tree, 
  data = automobile, 
  importance = TRUE, 
  ntree = 400, 
  mtry = 7,
  nodesize = 0.001,
  classwt = classwt,
  na.action = na.omit
)
classifiedforest_1
importance(classifiedforest_1)
varImpPlot(classifiedforest_1)

## Visualize with stargazer
oob_error <- classifiedforest_1$err.rate[classifiedforest_1$ntree, 1]
ntree <- classifiedforest_1$ntree
mtry <- classifiedforest_1$mtry
nodesize <- 0.001

importance_df <- as.data.frame(importance(classifiedforest_1))
top_features <- head(importance_df[order(-importance_df$MeanDecreaseAccuracy), , drop = FALSE], 5)

summary_table <- data.frame(
  Metric = c("OOB Error Rate", "Number of Trees (ntree)", "Number of Features (mtry)", "Min Node Size (nodesize)"),
  Value = c(round(oob_error, 4), ntree, mtry, nodesize)
)
stargazer(
  summary_table,
  type = "html",
  summary = FALSE,
  title = "Summary of Random Forest: After LDA Feature Selection",
  rownames = FALSE
)
stargazer(
  top_features,
  type = "html",
  title = "Top Features by Importance",
  summary = FALSE,
  rownames = TRUE
)

## Test without feature selection through lda
numerical_col <- automobile[, c(10,11,12,13,14,16,17,19:30)]
categorical_col <- automobile[, c(3:9,15,18)]
combined_cols <- c(names(categorical_col), names(numerical_col))
formula_tree <- as.formula(paste("symboling ~", paste(combined_cols, collapse="+")))
set.seed(1)
classifiedforest_2 <- randomForest(
  formula_tree, 
  data = automobile, 
  importance = TRUE, 
  ntree = 1800, 
  mtry = 7,
  nodesize = 0.001,
  classwt = classwt,
  na.action = na.omit
)
classifiedforest_2
importance(classifiedforest_2)
varImpPlot(classifiedforest_2)

oob_error <- classifiedforest_2$err.rate[classifiedforest_2$ntree, 1]
ntree <- classifiedforest_2$ntree
mtry <- classifiedforest_2$mtry
nodesize <- 0.001

summary_table <- data.frame(
  Metric = c("OOB Error Rate", "Number of Trees (ntree)", "Number of Features (mtry)", "Min Node Size (nodesize)"),
  Value = c(round(oob_error, 4), ntree, mtry, nodesize)
)
stargazer(
  summary_table,
  type = "html",
  summary = FALSE,
  title = "Summary of Random Forest: Without LDA Feature Selection",
  rownames = FALSE
)

## tuning
# ntree_vals <- c(500, 800, 1000, 1500, 1800, 2000)
# mtry_vals <- c(5, 6, 7, 8, 9, 10, 11)
# nodesize_vals <- c(0.001, 0.0001, 0.00001, 0.000001)
# 
# tuning_results <- data.frame(ntree = integer(),
#                              mtry = integer(),
#                              nodesize = numeric(),
#                              OOB_error = numeric())
# 
# for (ntree in ntree_vals) {
#   for (mtry in mtry_vals) {
#     for (nodesize in nodesize_vals) {
#       set.seed(1)
#       # Fit the random forest model
#       rf_model <- randomForest(
#         formula_tree,
#         data = automobile,
#         ntree = ntree,
#         mtry = mtry,
#         nodesize = nodesize,
#         classwt = classwt,
#         na.action = na.omit
#       )
# 
#       # Extract OOB error
#       oob_error <- rf_model$err.rate[ntree, "OOB"]
# 
#       # Append results
#       new_row <- data.frame(
#         ntree = ntree,
#         mtry = mtry,
#         nodesize = nodesize,
#         OOB_error = oob_error
#       )
#       tuning_results <- rbind(tuning_results, new_row)
#     }
#   }
# }
# best_params <- tuning_results[which.min(tuning_results$OOB_error), ]
# best_params

## Model interpretation
ggplot(automobile, aes(x = reorder(make, -symboling_numeric, FUN = mean), y = symboling_numeric)) +
  geom_boxplot(fill = "lightblue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Distribution of Symboling by Car Make") +
  ylab("Risk (Symboling)") +
  xlab("Car Make")

