# Red wine quality prediction through machine learning project
# edX HarvardX: PH125.9x - Data Science: Capstone
# Author: Priscila Trevino Aguilar
# Date: December 2020



##### Required packages installation #####
if(!require(caret))install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dyplr))install.packages("dyplr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2))install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(rpart))install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot))install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(randomForest))install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(Rborist))install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(klaR))install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(ggsci))install.packages("ggsci", repos = "http://cran.us.r-project.org")
if(!require(mlbench))install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(gapminder))install.packages("gapminder", repos = "http://cran.us.r-project.org")
if(!require(forcats))install.packages("forcats", repos = "http://cran.us.r-project.org")
if(!require(pROC))install.packages("forcats", repos = "http://cran.us.r-project.org")
if(!require(ggthemes))install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(corrplot))install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(e1071))install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(MLeval))install.packages("MLeval", repos = "http://cran.us.r-project.org")
if(!require(knitr))install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra))install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(params))install.packages("params", repos = "http://cran.us.r-project.org")




##### Required libraries loading #####
library(caret)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(Rborist)
library(klaR)
library(ggsci)
library(mlbench)
library(gapminder)
library(forcats)
library(pROC)
library(ggthemes)
library(corrplot)
library(e1071)
library(MLeval)
library(knitr)
library(gridExtra)
library(params)



##### Data loading #####
wine <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"), 
                 header = TRUE, 
                 sep = ";")




# Methods section #

##### Exploratory data analysis #####

## Getting acquainted with the data set
str(wine, width=80, strict.width="cut")
summary(wine)
sapply(wine, class) 
any(is.na(wine))  # no NA values verification


## Data visualization

# Wine quality rating histogram
wine %>% 
  ggplot(aes(quality)) + 
  geom_histogram(binwidth = 0.25, color = "#B22222", fill = "#8B0000") + 
  theme_economist() + 
  xlab("Quality") +
  ylab("Wine count") + 
  ggtitle("Wine quality rating distribution") + 
  theme(plot.title = element_text(hjust = 0.5))
      
# Measures of central tendency of wine quality
mean(wine$quality)
median(wine$quality)
sd(wine$quality)


# Volatile acidity histogram and quality per volatile acidity plot

va_wineplot <- wine %>%  # volatile acidity distribution
  ggplot(aes(volatile.acidity)) + 
  geom_histogram(binwidth = 0.25, color = "#B22222", fill = "#8B0000") + 
  theme_economist() + 
  xlab("Volatile acidity") +
  ylab("Wine count") + 
  ggtitle("Volatile acidity distribution") + 
  theme(plot.title = element_text(hjust = 0.5))

vaq_wineplot <- wine %>%  #quality per volatile acidity
  ggplot(aes(x = quality, y = volatile.acidity, color = quality)) + 
  geom_point(color = "#8B0000") + 
  theme_economist() +  
  xlab("Quality") +
  ylab("Volatile acidity") + 
  ggtitle("Quality distribution per volatile acidity") + 
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(va_wineplot, vaq_wineplot, nrow = 2)  # grid arrange 


# Alcohol content histogram and quality per alcohol content plot

ac_wineplot <- wine %>% # alcohol content distribution
  ggplot(aes(alcohol)) + 
  geom_histogram(binwidth = 0.25, color = "#B22222", fill = "#8B0000") +
  theme_economist() + 
  xlab("Alcohol content") +
  ylab("Wine count") + 
  ggtitle("Alcohol content distribution") + 
  theme(plot.title = element_text(hjust = 0.5))

acq_wineplot <- wine %>%  # quality per alcohol content
  ggplot(aes(x = quality, y = alcohol, color = quality)) + 
  geom_point(color = "#8B0000") + 
  theme_economist() +  
  xlab("Quality") +
  ylab("Alcohol content") + 
  ggtitle("Quality distribution per alcohol content") + 
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(ac_wineplot, acq_wineplot, nrow = 2) # grid arrange


# Wine category strip plot
category_wineplot <- wine %>%
                  mutate(category = cut(quality,
                  breaks = c(-Inf,5,Inf),            # the integer output variable was transformed...
                  labels = c("Moderate", "High")))   # ...into a dichotomous categorical variable
                
ggplot(category_wineplot, aes(x = category, y = quality, color = category)) +
  geom_jitter() +
  theme_economist() +
  scale_color_uchicago() +
  xlab("Category") +
  ylab("Quality rating") + 
  ggtitle("Wine category distribution") + 
  theme(legend.position = "none",
  plot.title = element_text(hjust = 0.5))

sum(category_wineplot$category == "Moderate") # number of wines in each category
sum(category_wineplot$category == "High")

# Multivariate plot of wine category by volatile acidity and alcohol content
ggplot(category_wineplot, aes(x = alcohol, y = volatile.acidity, color = category)) + 
  geom_point() + 
  theme_economist() +
  scale_color_uchicago() + 
  ggtitle("Category by volatile acidity and alcohol content") +
  xlab("Alcohol content") + 
  ylab("Volatile acidity") + 
  theme(legend.position = "right", 
        legend.text = element_text(size=10),
        title = element_text(size=8),
        plot.title = element_text(hjust = 0.5))




##### Data cleaning #####

# Correlation matrix and plot to identify correlated variables
correlationMatrix <- cor(wine[,1:11])

corrplot(correlationMatrix,  # correlation plot     
          order = "hclust", addrect = 7, 
          col = c("darkgray", "darkred"), 
          bg = "azure2") 

# Removal of highly correlated variables 
correlated <- findCorrelation(correlationMatrix, cutoff = 0.5)  # the cutoff was set
print(correlated)

correlationMatrix[,c(1,3,7)]  # identified highly correlated attributes by their index 

wine_clean <- select(wine, -c(1,3,7))  # correlated variables removal




##### Regression models approach #####

# Regression models' loss function (Root Square Mean Error)
RMSE <- function(actual, predicted){
  sqrt(mean((actual - predicted)^2))
}

# Data partition and train and test set creation for regression models
set.seed(6, sample.kind="Rounding")
test_index <- createDataPartition(y = wine_clean$quality, times = 1, p = 0.2, 
                                  list = FALSE)

train_reg <- wine_clean[-test_index,]
test_reg <- wine_clean[test_index,]



## Regression tree model

reg_tree <- rpart(quality ~ .,       # regression tree fit 
                  data = train_reg, 
                  method = "anova")

# The model automatically applied a range of cost complexity and performed cross validation
reg_tree$cptable
plotcp(reg_tree)

# Further model improvement with hyper parameter tuning through grid search (cross validation)
tuning_grid <- expand.grid(minsplit = seq(3, 20, 1), maxdepth = seq(5, 15, 1))  # tuning grid
nrow(tuning_grid)  # number of combinations in tuning grid


model_comb <- list()    
for (i in 1:nrow(tuning_grid)) {  # for loop to automate model training combinations 
  minsplit <- tuning_grid$minsplit[i]
  maxdepth <- tuning_grid$maxdepth[i]
  
  model_comb[[i]] <- rpart(quality ~ .,
    train_reg,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}


opt_cp <- function(x) {            # function to obtain the optimal cp
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}


min_error <- function(x) {          # function to obtain the minimum error
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}



tuning <- tuning_grid %>%   # grid search results
       mutate(cp = purrr::map_dbl(model_comb, opt_cp), 
       error = purrr::map_dbl(model_comb, min_error)) %>%
       arrange(error) %>%
       top_n(-5, wt = error)


tuning[1,]  # optimal parameters
tuning[1,3]  # optimal cp


# Pruned regression tree with the optimal cp
pruned_tree <- prune(reg_tree, cp = 0.01)


# Variable importance plot
Importance <- as.numeric(pruned_tree$variable.importance[1:5])
Variable <- c("Alcohol", "Sulphates", "Volatile acidity", "Density", "Chlorides")
treevar_plot <- data.frame(Variable, sort(Importance))  # data frame to create plot

treevar_plot %>% ggplot(aes(x = reorder(Variable, Importance),  
                       y = Importance, fill = Variable)) + 
                       geom_bar(stat = "identity", width = .6) + 
                       scale_fill_lancet() + 
                       coord_flip() + 
                       theme_economist() + 
                       ggtitle("Variable importance in wine quality") + 
                       theme(axis.title.y = element_blank(),
                       plot.title = element_text(size=12),
                       legend.title = element_blank(),
                       legend.position = "none")

# Rpart tree plot
rpart.plot(pruned_tree, type = 3, digits = 3, fallen.leaves = TRUE)

# Adjusted tree Plot
prp(pruned_tree, 
    fallen.leaves = TRUE, 
    branch = .5,
    faclen = 3,
    shadow.col = "darkcyan", 
    branch.lty = 3, 
    split.cex = 1, 
    split.prefix = "is ", 
    split.suffix = "?", 
    split.box.col = "azure2", 
    split.border.col = "darkcyan",
    split.round = .5)


# Regression tree model evaluation through RMSE
regtree_pred <- predict(pruned_tree, test_reg)  # computed predictions
regtree_rmse <- RMSE(test_reg$quality, regtree_pred)  # computed and saved RMSE
print(regtree_rmse)




## Random forest regression model

reg_forest <- randomForest(quality ~ .,  # random forest regression fit 
                       data = train_reg, 
                       type = "regression", 
                       proximity = TRUE)

# Random forest variable importance plot
varImpPlot(reg_forest)

# Cross validation to obtain optimal parameters
regforest_cv <- train(quality ~ .,
      method = "Rborist",
      tuneGrid = data.frame(predFixed = 3, minNode = c(3, 50)),
      data = train_reg)

# Model evaluation through RMSE
regforest_pred <- predict(regforest_cv, test_reg)
rf_rmse <- RMSE(test_reg$quality, regforest_pred)
print(rf_rmse)




##### Classification models approach #####

# Dichotomization of the integer output variable into moderate quality and high quality 
category <- cut(wine_clean$quality,
           breaks = as.character(c(-Inf, 5, Inf)),  # cutoff was set
           labels = c("Moderate", "High"), 
           include.lowest = TRUE)


# Data set for classification models
class_set <- wine_clean %>% mutate(category = category) %>% select(-9) 
head(class_set)  # the new category column was added

# Data partition and train and test sets for the classification models
set.seed(6, sample.kind="Rounding")
test_index <- createDataPartition(y = class_set$category, times = 1, p = 0.2,  
                               list = FALSE)

train_class <- class_set[-test_index,]  
test_class <- class_set[test_index,]

# Computations control for cross validation with the train function 
models_control <- trainControl(method = "repeatedcv", 
                               number = 10, 
                               repeats = 3, 
                               classProbs = TRUE, 
                               savePredictions = TRUE,
                               summaryFunction = twoClassSummary)




## Naive Bayes classification model 

bayes_class <- train(category~.,  # Naive Bayes fit 
              data = train_class,
              method = "nb",
              metric = "ROC", 
              trControl = models_control)  # cross validation

# Model results
bayes_class$results
bayes_class$bestTune

# Variable importance plot
temp_nb <- varImp(bayes_class)
plot(temp_nb)

# Model evaluation through confusion matrix and accuracy computation
nb_accuracy <- confusionMatrix(predict(bayes_class, test_class), 
                               test_class$category)$overall["Accuracy"]
print(nb_accuracy)




## Random forest classification model

forest_class <- train(category~ .,        # random forest classification fit
                      data = train_class,
                      method = "rf", 
                      metric = "ROC",
                      trControl = models_control,
                      tunegrid = expand.grid(mtry = c(1:15)))  # cross validation

# Variable importance plot
temp_rf <- varImp(forest_class)
plot(temp_rf)

# Plot of a single classification tree
class_treeplot <- rpart(category~ ., 
                   data = train_class, 
                   method = "class")
prp(class_treeplot,
    fallen.leaves = TRUE, 
    branch = .5,
    faclen = 3,
    shadow.col = "darkcyan", 
    branch.lty = 3, 
    split.cex = 1, 
    split.prefix = "is ", 
    split.suffix = "?", 
    split.box.col = "azure2", 
    split.border.col = "darkcyan",
    split.round = .5)

# Model evaluation through confusion matrix and accuracy computation
rf_accuracy <- confusionMatrix(predict(forest_class, test_class),
                test_class$category)$overall["Accuracy"]
print(rf_accuracy)



# Classification models ROC curves
models <- list(bayes_class, forest_class)
plot_roc <- evalm(models,    # ROC curve plot
                  plots='r',
                  gnames = c("Naive Bayes", "Random Forest"), 
                  title = "Classification models ROC curves")


# Results section #

##### Results tables #####
reg_results <- data.frame(Model = c("Regression tree", "Random Forest Regression"), 
                   RMSE = c(regtree_rmse, rf_rmse))
kable(reg_results, caption = "Results of Regression Analysis")  # regression results
  
class_results <- data.frame(Model = c("Naive Bayes", "Random Forest Classification"), 
                              Accuracy = c(nb_accuracy, rf_accuracy))
kable(class_results, caption = "Results of Classification Analysis")  # classification results


# Appendix #

# Environment
print("Operating System:")
version

