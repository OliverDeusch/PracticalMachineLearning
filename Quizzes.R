########
# week 1
########

Answers in PDF.


########
# week 2
########

Week 2 -- Question 1
####################

1. Load the Alzheimer's disease data using the commands:

library(AppliedPredictiveModeling)
data(AlzheimerDisease)

Which of the following commands will create non-overlapping training and test sets with about 50% of the observations assigned to each?

adData = data.frame(diagnosis,predictors)
testIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[-testIndex,]
testing = adData[testIndex,]

adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis,p=0.5,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

#adData = data.frame(diagnosis,predictors)
#train = createDataPartition(diagnosis, p = 0.50,list=FALSE)
#test = createDataPartition(diagnosis, p = 0.50,list=FALSE)

#adData = data.frame(diagnosis,predictors)
#trainIndex = createDataPartition(diagnosis,p=0.5,list=FALSE)
#training = adData[trainIndex,]
#testing = adData[trainIndex,]


Week 2 -- Question 2
####################

2. Load the cement data using the commands:

library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

Make a plot of the outcome (CompressiveStrength) versus the index of the samples. Color by each of the variables in the data set
(you may find the cut2() function in the Hmisc package useful for turning continuous covariates into factors). What do you notice
in these plots?

library(Hmisc)
par(mfrow = c(3,3))
plot(concrete$CompressiveStrength, col=cut2(concrete$FlyAsh), main="FlyAsh")
plot(concrete$CompressiveStrength, col=cut2(concrete$Age), main="Age")
plot(concrete$CompressiveStrength, col=cut2(concrete$BlastFurnaceSlag), main="BlastFurnaceSlag")
plot(concrete$CompressiveStrength, col=cut2(concrete$Water), main="Water")
plot(concrete$CompressiveStrength, col=cut2(concrete$Superplasticizer), main="Superplasticizer")
plot(concrete$CompressiveStrength, col=cut2(concrete$CoarseAggregate), main="CoarseAggregate")
plot(concrete$CompressiveStrength, col=cut2(concrete$FineAggregate), main="FineAggregate")
plot(concrete$CompressiveStrength, col=cut2(concrete$Cement), main="Cement")



 There is a non-random pattern in the plot of the outcome versus index.
 There is a non-random pattern in the plot of the outcome versus index that is perfectly explained by the FlyAsh variable so there may be a variable missing.
 There is a non-random pattern in the plot of the outcome versus index that is perfectly explained by the Age variable so there may be a variable missing.
*There is a non-random pattern in the plot of the outcome versus index that does not appear to be perfectly explained by any predictor suggesting a variable may be missing.




Week 2 -- Question 3
####################

3. Load the cement data using the commands:

library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

Make a histogram and confirm the Superlasticizer variable is skewed.
Normally you might use the log transform to try to make the data more symmetric.
Why would that be a poor choice for this variable?

par(mfrow = c(1,1))
hist(concrete$Superplasticizer)
summary(concrete$Superplasticizer)
hist(log(concrete$Superplasticizer + 1))

*There are a large number of values that are the same and even if you took the log(SuperPlasticizer + 1) they would still all be identical so the distribution would not be symmetric.
 The log transform does not reduce the skewness of the non-zero values of SuperPlasticizer
 The SuperPlasticizer data include negative values so the log transform can not be performed.
 The log transform is not a monotone transformation of the data.



Week 2 -- Question 4
####################

4. Load the Alzheimer's disease data using the commands:
  
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

Find all the predictor variables in the training set that begin with IL.
Perform principal components on these variables with the preProcess() function from the caret package.
Calculate the number of principal components needed to capture 90% of the variance. How many are there?


names(training)
names(training)[grep("^IL", names(training))]

preProcess(training[, grep("^IL", names(training))], method = "pca", thresh = 0.90)

PCA needed 9 components to capture 90 percent of the variance



*9
 10
 8
 7


Week 2 -- Question 5
####################

5. Load the Alzheimer's disease data using the commands:

library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

Create a training data set consisting of only the predictors with variable names beginning with IL and the diagnosis.
Build two predictive models, one using the predictors as they are and one using PCA with principal components explaining
80% of the variance in the predictors. Use method="glm" in the train function.

What is the accuracy of each method in the test set? Which is more accurate?

training = adData[ inTrain, c(1, grep("^IL", names(adData)))]
testing = adData[-inTrain, c(1, grep("^IL", names(adData)))]

modelFit <- train(training$diagnosis ~ .,method="glm", data=training)
confusionMatrix(testing$diagnosis,predict(modelFit,testing))

               Accuracy : 0.6463



preProc <- preProcess(training[, -1], method="pca", thresh = 0.8)
trainPC <- predict(preProc, training[, -1])
modelFit <- train(training$diagnosis ~ .,method="glm",data=trainPC)

testPC <- predict(preProc, testing[, -1])
confusionMatrix(testing$diagnosis, predict(modelFit,testPC))

               Accuracy : 0.7195   



*
Non-PCA Accuracy: 0.65
PCA Accuracy: 0.72




########
# week 3
########


Week 3 -- Question 1
####################

1. For this quiz we will be using several R packages. R package versions change over time, the right answers have been checked using the following versions of the packages.

AppliedPredictiveModeling: v1.1.6

caret: v6.0.47

ElemStatLearn: v2012.04-0

pgmm: v1.1

rpart: v4.1.8

If you aren't using these versions of the packages, your answers may not exactly match the right answer,
but hopefully should be close. Load the cell segmentation data from the AppliedPredictiveModeling package using the commands:
  
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

1. Subset the data to a training set and testing set based on the Case variable in the data set.
2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings.
3. In the final model what would be the final model prediction for cases with the following variable values:

set.seed(125)
train <- segmentationOriginal[segmentationOriginal$Case=="Train", -2]
test <- segmentationOriginal[segmentationOriginal$Case=="Test", -2]

modelFit <- train(train$Class ~ .,method="rpart",data=train)

modelFit$finalModel

library(rpart)
library(ggplot2)
library(rattle)
library(rpart.plot)
fancyRpartPlot(modelFit$finalModel)


a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2
b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100
c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100
d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2

a. PS
b. WS
c. PS
d. Not possible to predict



Week 3 -- Question 2
####################

2. If K is small in a K-fold cross validation is the bias in the estimate of out-of-sample (test set)
accuracy smaller or bigger? If K is small is the variance in the estimate of out-of-sample (test set)
accuracy smaller or bigger. Is K large or small in leave one out cross validation?

The bias is larger and the variance is smaller. Under leave one out cross validation K is equal to the sample size.


Week 3 -- Question 3
####################

3. Load the olive oil data using the commands:
   
library(pgmm)
data(olive)
olive = olive[,-1]

(NOTE: If you have trouble installing the pgmm package, you can download the -code-olive-/code- dataset here: olive_data.zip. After unzipping the archive, you can load the file using the -code-load()-/code- function in R.)
 
These data contain information on 572 different Italian olive oils from multiple regions in Italy.
Fit a classification tree where Area is the outcome variable. Then predict the value of area for the
following data frame using the tree command with all defaults


modelFit <- train(olive$Area ~ .,method="rpart",data=olive)
modelFit$finalModel

newdata = as.data.frame(t(colMeans(olive)))
predict(modelFit$finalModel, newdata)
1 
2.783282 

What is the resulting prediction? Is the resulting prediction strange? Why or why not?
 
  4.59965. There is no reason why the result is strange.
  0.005291005 0 0.994709 0 0 0 0 0 0. There is no reason why the result is strange.
  0.005291005 0 0.994709 0 0 0 0 0 0. The result is strange because Area is a numeric variable and we should get the average within each leaf.
* 2.783. It is strange because Area should be a qualitative variable - but tree is reporting the average value of Area as a numeric variable in the leaf predicted for newdata

 
 
Week 3 -- Question 4
####################

4. Load the South Africa Heart Disease Data and create training and test sets with the following code:
  
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

Then set the seed to 13234 and fit a logistic regression model
(method="glm", be sure to specify family="binomial") with Coronary Heart Disease (chd)
as the outcome and age at onset, current alcohol consumption, obesity levels, cumulative tabacco,
type-A behavior, and low density lipoprotein cholesterol as predictors. Calculate the
misclassification rate for your model using this function and a prediction on the "response" scale:

set.seed(1234)
model <- train(chd~age+alcohol+obesity+tobacco+typea+ldl,data=trainSA,method="glm",family="binomial")

missClass <- function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

missClass(trainSA$chd, predict(model, trainSA))
[1] 0.2727273
missClass(testSA$chd, predict(model, testSA))
[1] 0.3116883
  
What is the misclassification rate on the training set? What is the misclassification rate on the test set?


Test Set Misclassification: 0.31
Training Set: 0.27


Week 3 -- Question 5
####################

Load the vowel.train and vowel.test data sets:

#install.packages("ElemStatLearn")
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

Set the variable y to be a factor variable in both the training and test set.
vowel.train$y <- factor(vowel.train$y)
vowel.test$y <- factor(vowel.test$y)

Then set the seed to 33833.
set.seed(33833)

Fit a random forest predictor relating the factor variable y to the remaining variables.
Read about variable importance in random forests here:
http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
The caret package uses by default the Gini importance.

modFit <- train(y ~ . ,data=vowel.train, method="rf")
modFit

Calculate the variable importance using the varImp function in the caret package.
varImp(modFit)

What is the order of variable importance?


The order of the variables is:
  
  x.10, x.7, x.9, x.5, x.8, x.4, x.6, x.3, x.1,x.2

(closest but not quite right)
The order of the variables is:
  
  x.2, x.1, x.5, x.6, x.8, x.4, x.9, x.3, x.7,x.10

The order of the variables is:
  
  x.1, x.2, x.3, x.8, x.6, x.4, x.5, x.9, x.7,x.10

The order of the variables is:
  
  x.10, x.7, x.5, x.6, x.8, x.4, x.9, x.3, x.1,x.2




########
# week 4
########


Week 4 -- Question 1
####################

1. For this quiz we will be using several R packages. R package versions change over time, the right answers have been checked
using the following versions of the packages.

AppliedPredictiveModeling: v1.1.6

caret: v6.0.47

ElemStatLearn: v2012.04-0

pgmm: v1.1

rpart: v4.1.8

gbm: v2.1

lubridate: v1.3.3

forecast: v5.6

e1071: v1.6.4

If you aren't using these versions of the packages, your answers may not exactly match the right answer, but hopefully should be close.

Load the vowel.train and vowel.test data sets:

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

Set the variable y to be a factor variable in both the training and test set. Then set the seed to 33833.

vowel.train$y <- factor(vowel.train$y)
vowel.test$y <- factor(vowel.test$y)
set.seed(33833)

Fit (1) a random forest predictor relating the factor variable y to the remaining variables and (2) a boosted predictor using the "gbm" method.
Fit these both with the train() command in the caret package.

library(caret)
fit_rf <- train(y ~ . , data = vowel.train, method = "rf")
fit_gbm <- train(y ~ . , data = vowel.train, method = "gbm")

pred_rf <- predict(fit_rf, vowel.test)
pred_gbm <- predict(fit_gbm, vowel.test)

confusionMatrix(pred_rf, vowel.test$y)$overall[1]
Accuracy 
0.5995671 
confusionMatrix(pred_gbm, vowel.test$y)$overall[1]
Accuracy 
0.525974

pred_both <- data.frame(pred_rf, pred_gbm, y = vowel.test$y)

sum(pred_rf[pred_both$pred_rf == pred_both$pred_gbm] == 
        pred_both$y[pred_both$pred_rf == pred_both$pred_gbm]) / 
    sum(pred_both$pred_rf == pred_both$pred_gbm)
[1] 0.6386293

What are the accuracies for the two approaches on the test data set? What is the accuracy among the test set samples where the two methods agree?


RF Accuracy = 0.9987
GBM Accuracy = 0.5152
Agreement Accuracy = 0.9985

*
RF Accuracy = 0.6082
GBM Accuracy = 0.5152
Agreement Accuracy = 0.6361

RF Accuracy = 0.3233
GBM Accuracy = 0.8371
Agreement Accuracy = 0.9983

RF Accuracy = 0.6082
GBM Accuracy = 0.5152
Agreement Accuracy = 0.5325



Week 4 -- Question 2
####################

2. Load the Alzheimer's data using the following commands

library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

Set the seed to 62433 and predict diagnosis with all the other variables using a random forest ("rf"), boosted trees ("gbm")
and linear discriminant analysis ("lda") model. Stack the predictions together using random forests ("rf").

set.seed(62433)
fit_rf <- train(diagnosis ~ . , data = training, method = "rf")
fit_gbm <- train(diagnosis ~ . , data = training, method = "gbm")
fit_lda <- train(diagnosis ~ . , data = training, method = "lda")
pred_rf <- predict(fit_rf, testing)
pred_gbm <- predict(fit_gbm, testing)
pred_lda <- predict(fit_lda, testing)

pred_all <- data.frame(pred_rf, pred_gbm, pred_lda, diagnosis = testing$diagnosis)

fit_comb <- train(diagnosis ~ . , method = "rf", data = pred_all)
pred_comb <- predict(fit_comb, pred_all)



confusionMatrix(pred_rf, testing$diagnosis)$overall[1]
Accuracy 
0.7804878 
confusionMatrix(pred_gbm, testing$diagnosis)$overall[1]
Accuracy 
0.804878
confusionMatrix(pred_lda, testing$diagnosis)$overall[1]
Accuracy 
0.7682927
confusionMatrix(pred_comb, testing$diagnosis)$overall[1]
Accuracy 
0.8170732
# This is stacked accuracy

What is the resulting accuracy on the test set? Is it better or worse than each of the individual predictions?


  Stacked Accuracy: 0.76 is better than random forests and boosting, but not lda.
  Stacked Accuracy: 0.80 is better than all three other methods
* Stacked Accuracy: 0.80 is better than random forests and lda and the same as boosting.
  Stacked Accuracy: 0.76 is better than lda but not random forests or boosting.


Week 4 -- Question 3
####################

3. Load the concrete data with the commands:
  
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

Set the seed to 233 and fit a lasso model to predict Compressive Strength.

set.seed(233)
library(elasticnet)
mod_lasso <- train(CompressiveStrength ~ . , data = training, method = "lasso")
plot.enet(mod_lasso$finalModel, xvar = "penalty", use.color = TRUE)

Which variable is the last coefficient to be set to zero as the penalty increases? (Hint: it may be useful to look up ?plot.enet).

  FineAggregate
* Cement
  BlastFurnaceSlag
  CoarseAggregate

  

Week 4 -- Question 4
####################

4. Load the data on the number of visitors to the instructors blog from here:

  https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv
  
Using the commands:

setwd("C:/Users/deuscoli/Desktop/Coursera_DataScience/8_Practical_Machine_Learning/")
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv"
download.file(fileUrl, destfile = "gaData.csv", method = "curl")    
list.files()

library(lubridate) # For year() function below

dat = read.csv("gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

Fit a model using the bats() function in the forecast package to the training time series. Then forecast this model for the remaining time points.

install.packages("forecast")
library(forecast)
fit_ts <- bats(tstrain)
fcast <- forecast(fit_ts, level = 95, h = dim(testing)[1])
sum(fcast$lower < testing$visitsTumblr & testing$visitsTumblr < fcast$upper) / dim(testing)[1]


For how many of the testing points is the true value within the 95% prediction interval bounds?

  93%
* 96%
  94%
  95%


Week 4 -- Question 5
####################

5. Load the concrete data with the commands:
  
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

  
Set the seed to 325 and fit a support vector machine using the e1071 package to predict Compressive Strength using the default settings.

set.seed(325)
library(e1071)
fit_svm <- svm(CompressiveStrength ~ . , data = training)
pred_svm <- predict(fit_svm, testing)
accuracy(pred_svm, testing$CompressiveStrength)
ME     RMSE      MAE       MPE     MAPE
Test set 0.1682863 6.715009 5.120835 -7.102348 19.27739

Predict on the testing set. What is the RMSE?

  45.09
  11543.39
* 6.72
  6.93

