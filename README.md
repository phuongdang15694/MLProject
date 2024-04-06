Comparison of ML Classification Models for Flu Prediction
================
 
*April 2024*
## Introduction
In recent year, public health research has been learning about understanding the underlying causes that influence how infectious diseases spread, including the common flu. This report focuses on analyzing biologic characteristics as well as lifestyle and behavioral factors that might impact on the likelihood of adults catching the common flu from December 1 to March 31. These variables include:

- Exercise: Estimated weekly hours of physical activity

- Sugar Consumption: The estimated amount of sugar consumed per day

- Work Hours: The estimated weekly average number of hours worked

- Handwashing Frequency: The number of times an adult reports washing their hands each day.

- Body Mass Index (BMI): A person’s BMI as of December 1.

- Age: The individual’s age.

- Smoking Status: Indicates whether the individual smokes, including both traditional smoking and vaping

- Flu Incidence (Response Variable): Whether the person contracted the common flu, which is the primary outcome of interest

The purpose of this study is to establish and evaluate distinct classification models to predict flu incidence using the factors mentioned above. This study employs <strong>*Logistic regression*</strong>, <strong>*Linear discriminant analysis*</strong>, <strong>*Quadratic discriminant analysis*</strong>, and <strong>*Gaussian naive Bayesian* </strong> classifiers to identify important factors that can predict the flu then assess how well each model does so. The findings of this study may be useful in interventions that change controllable behaviours and provide advice to people who are more likely to contract the flu.

In the following sections, we will detail the methodology employed in cleaning the data, building the predictive models, and evaluating their performance. The report will conclude with a summary of the findings and recommendations based on the analysis.
## Preliminary Analysis and Data Cleaning
### Data Completeness and Accuracy Verification
The initial step in data analysis involved preprocessing and cleaning thoroughly the data. Our review revealed that the dataset was mostly complete, with a few exceptions in specific variables that needed attention. We discovered that there are several “0” values in some features. Although this error is acceptable in other features, it is illogical in BMI. Therefore, we decided to replace these anomalous values with 29 (which appears to be the approximate mean and median of BMI calculated from the remaining entries in the dataset). This approach ensured that our subsequent analysis would not be skewed by these outliers. Regarding other predictors’ anomalies, we have no proof whether they are inaccurate or not, so we decided to maintain their original values.
### Data Import and Preparation
The dataset then was splitted into 2 distinct sets, which are training and testing set, with a ratio of 7:3 respectively.
```{r}
set.seed(123) 
split_data <- sample(c(TRUE, FALSE), nrow(flu_data),replace = TRUE, prob = c(0.7,0.3))
flu_train <- flu_data[split_data, ]
flu_test <- flu_data[!split_data, ]
flu_test_noresponse = subset(flu_test,select=-flu)
```
ROC curves plot the
True positive rate  (*sensitivity*) vs. False positives  (*1-specificity)
for different values of c.

Before we begin to train the dataset, we probably want to look at which predictors / factors are likely to be effective. We’ll use the `featurePlot()` function from the `caret()` library

Model Building

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/c5d46cde-6635-43ed-88f4-f684d7b32ebc)

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/942ec7fd-0564-4010-8310-a801b6356a4e)

In this set of graphs, we’re looking at class distributions of all 7 predictors by the response Flu.
- In the sugar panel, following by bmi panel, there are just minor separation or boundaries.
- In the other panels, we are unable to identify the significant differences.

## Linear Discriminant Analysis (LDA)
LDA is used find a linear combination of features that characterizes or
separate two or more classes of objects or events

We use function `lda()` (from package `MASS`)
```{r }
#Train on training set with LDA model
lda_train = lda(flu~. , data=flu_train)
lda_train #LDA model

#Apply on the testing set
lda_predict = predict(lda_train, flu_test_noresponse)

#Measure how well the model performs on the test set
lda.cm = table(flu_test$flu,lda_predict$class)
lda_metrics=confusionMatrix(lda.cm,positive="1")
lda_roc = roc(flu_test$flu, lda_predict$posterior[,2])

```
Prob of Flu and Non-Flu by LDA

The class-specific prior is simply the proportion of data points that
belong to the class. The class-specific mean vector is the average of
the input variables that belong to the class. 

```{r}
gt(data.frame("Flu"=round(lda_train$prior["1"],4),"No Flu"= round(lda_train$prior["0"],4)))%>%tab_header("Prior probability")
```

| Flu     | No Flu | 
| :-----: | :----: | 
| 0.335   |  0.665 | 

Table: Prior probability

The linear combination of Exercise, Sugar, Work, Handwash, BMI, Age, and Smoking=Yes that are used to form the LDA decision rule, which is:

log()0.062xExercise + 0.05xSugar - 0.019xWork - 0.035xHandwash + 0.091xBMI + 0.02xAge + 0.375xSmoking

The `plot()` function can be used to used to plot the linear discriminants
using the equation above for each of the training observations.

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/1ab3eef4-8e24-4969-a3b1-0d52bb44ab0f)

Comment: The zero is the decision boundary.

## Quadratic Discriminant Analysis (QDA)
 Like LDA, Quadratic Discriminant Analysis (QDA) is another
generative model that assumes that each class follows a Gaussian distribution.

 The only difference is the class-specific variances are different.

We use function `qda()` (from package `MASS`) to train dataset with QDA

```{r }
#train dataset
qda_train = qda(flu~. , data=flu_train)

#predict with test set
qda_predict = predict(qda_train, flu_test_noresponse)

#Measure how well the model performs on the test set
qda.cm = table(flu_test$flu,qda_predict$class)
qda_metrics=confusionMatrix(qda.cm,positive="1")
qda_roc = roc(flu_test$flu, qda_predict$posterior[,2])
```

Prob of Flu and Non Flu 

```{r  include=FALSE, warning=FALSE, message = FALSE, comment=""}
gt(data.frame("Flu"=round(qda_train$prior["1"],4),"No Flu"= round(qda_train$prior["0"],4)))%>%tab_header("Prior probability")
```
| Flu     | No Flu | 
| :-----: | :----: | 
| 0.335   |  0.665 | 

Table: Prior probability

## Gaussian naive Bayesian classifier

