# Comparison of Machine Learning Classification Models for Common Flu Prediction
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

The purpose of this study is to establish and evaluate distinct classification models to predict flu incidence using the factors mentioned above. This study employs logistic regression, linear and quadratic discriminant analysis, and Gaussian naive Bayesian classifiers to identify important factors that can predict the flu then assess how well each model does so. The findings of this study may be useful in interventions that change controllable behaviours and provide advice to people who are more likely to contract the flu.

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

Model Building

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/c5d46cde-6635-43ed-88f4-f684d7b32ebc)

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/942ec7fd-0564-4010-8310-a801b6356a4e)

In this set of graphs, we’re looking at class distributions of all 7 predictors by the response Flu.
> In the sugar panel, following by bmi panel, there are just minor separation or boundaries.
> In the other panels, we are unable to identify the significant differences.

## LDA
We use function `lda` (from package `MASS`)
```{r }
lda_train = lda(flu~. , data=flu_train)
lda_train

lda_predict = predict(lda_train, flu_test_noresponse)
lda.cm = table(flu_test$flu,lda_predict$class)
lda_metrics=confusionMatrix(lda.cm,positive="1")

lda_roc = roc(flu_test$flu, lda_predict$posterior[,2])

```

| Flu     | No Flu | 
| :-----: | :----: | 
| 0.335   |  0.665 | 

Table: Prior probability

The linear combination of Exercise, Sugar, Work, Handwash, BMI, Age, and Smoking=Yes that are used to form the LDA decision rule, which is:

0.062xExercise + 0.05xSugar - 0.019xWork - 0.035xHandwash + 0.091xBMI + 0.02xAge + 0.375xSmoking

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/1ab3eef4-8e24-4969-a3b1-0d52bb44ab0f)



