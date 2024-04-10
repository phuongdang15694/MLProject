ML Classification Models Comparison for Flu Prediction 
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
The initial step in data analysis involved preprocessing and cleaning thoroughly the data. Our review revealed that the dataset was mostly complete, with a few exceptions in specific variables that needed attention. 

We discovered that there are several “0” values in some features. Although this error is acceptable in other features, it is illogical in BMI. Therefore, we decided to replace these anomalous values with 29 (which appears to be the approximate mean and median of BMI calculated from the remaining entries in the dataset). 

This approach ensured that our subsequent analysis would not be skewed by these outliers. Regarding other predictors’ anomalies, we have no proof whether they are inaccurate or not, so we decided to maintain their original values.
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
  
## Logistic regression
Logistic regression have mostly been used to estimate class probabilities.

We model πi using the log-odds ratio, also called the logit function: 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/32b57931-431d-4175-b715-29e5092f5157)

We use `glm` function  
```{r}
#train model
logit_train = glm(flu ~ ., data = flu_train, family = "binomial")
logit_predict = predict(logit_train,flu_test_noresponse,type = "response")
#Confusion Matrix
classify50 <- ifelse(logit_predict > 0.5, "1", "0")
cm= table(Predicted = classify50, Actual = flu_test$flu)
logit_metrics=confusionMatrix(cm,positive = "1")
#ROC
logit_roc = roc(flu_test$flu, logit_predict)
```
Logistic model: 
logit= -10.02224853 + 0.083xExercise +	0.064xSugar - 0.025xWork - 0.058xHandwash	+ 0.124xBMI +	0.026xAge + 0.448xSmoking

## Linear Discriminant Analysis (LDA)
- LDA is used find a linear combination of features that characterizes or
separate two or more classes of objects or events.

- LDA assumes that the covariance matrix across classes is the same.

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
=> QDA does not assume constant covariance matrix across classes

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

<strong> * Summary of LDA and QDA* </strong>

– Pros
* It is simple, fast and portable. Outperforms logistic regression
when its assumptions are met.

– Cons
* It requires normal distribution assumption on features/predictors.
* QDA can get computationally expensive with many predictors.
* QDA is that it cannot be used as a dimensional reduction technique.
  
## Gaussian naive Bayesian classifier
The Gaussian Naive Bayes classifier assumes x variables are independent within a class, implying diagonal covariance matrices.

– Pros

* It is easy and fast to predict class of test data set. It also perform
well in multi-class prediction.

* When assumption of independence holds, the NB classifier performs better than logistic regression with few data.
  
– Cons

* If a categorical variable has a category in the test set, but not observed in the training set, then the model will assign a zero probability and will be unable to make a prediction. If so, use a smoothing technique is called Laplace estimation.
  
* There is a strong set of assumptions on the distribution of the
features such as normal, multinomial etc.

* There is also the assumption of independence in predictors which
is quite rare.

We use naive_bayes() function.
```{r}
#Build model and predict with NB
nb_train =  naive_bayes(flu~. , data=flu_train)
nb_train
nb_predict = predict(nb_train,flu_test_noresponse)

#Implement Result
nb_metrics=confusionMatrix(factor(nb_predict),factor(flu_test$flu))
nb_predict_prob = predict(nb_train,flu_test_noresponse, type = "prob")
nb_roc = roc(flu_test$flu, nb_predict_prob[,2])
gt(data.frame("Flu"=round(nb_train$prior["1"],3),"No Flu"= round(nb_train$prior["0"],3)))%>%tab_header("Prior probability")
```
| Flu     | No Flu | 
| :-----: | :----: | 
| 0.335   |  0.665 | 

 ## Model Evaluation
```{r}
#Prep
cm_models = data.frame("Models"=c("LDA","QDA","Naive Bayes","Logistic"), 
                       "Accuracy"= round(c(lda_metrics$overall["Accuracy"],qda_metrics$overall["Accuracy"],
                                           nb_metrics$overall["Accuracy"],logit_metrics$overall["Accuracy"]),3),
                   "Sensitivity" = round(c(lda_metrics$byClass["Sensitivity"],qda_metrics$byClass["Sensitivity"],
                                            nb_metrics$byClass["Sensitivity"],logit_metrics$byClass["Sensitivity"]),3),
                   "Specificity" = round(c(lda_metrics$byClass["Specificity"],qda_metrics$byClass["Specificity"],
                                            nb_metrics$byClass["Specificity"],logit_metrics$byClass["Specificity"]),3)
                      )

cm_models %>%gt() %>% tab_header("Metric Comparison")
```

| Models     | Accuracy    |Sensitivity |Specificity |
| :-----:    | :----:      | :----:     | :----:     |
| LDA        |  0.750      | 0.691      | 0.775      | 
| QDA        |  0.706      | 0.603      | 0.760      | 
|Naive Bayes |  0.750      | 0.807      | 0.651      | 
| Logistic   |  0.741      | 0.566      | 0.841      | 

 Table: Metric Comparison
 
 *Comments*
 - Sensitivity here implies that the model can correctly identify most patients who have the disease. Therefore, we should choose Naive Bayes in scenarios where missing a positive case has serious consequences, such as screening at risk areas in epidemic.
 - On the other hand, specificity implies correctly recognizing individuals who are not at risk of contracting the flu. Thus, we should choose Logistic regression when false positives lead to significant costs, such as vaccines distribution - who really needs flu shot when there is a shortage.
 - Accuracy gives you an overall success rate, but it might not show how well the model really works.
   
 ```{r}
auc_models= data.frame ("Models"=c("LDA","QDA","NaiveBayes","Logistic"),
"AUC"=c(auc(lda_roc),auc(qda_roc),auc(nb_roc),auc(logit_roc)))
auc_models %>% gt() %>% tab_header("AUC Comparison")
```
| Models      | AUC        | 
| :-----:     | :----:     | 
| LDA         |  0.8162027 | 
| QDA         |  0.7911093 |
| Naive Bayes |  0.8089738 |
| Logistic    |  0.8149564 |

```{r}
ggroc(list(lda=lda_roc,qda = qda_roc,nb=nb_roc, logreg= logit_roc))+theme_bw()+ggtitle("ROC Curve")
```
![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/96ba06fa-44a3-4a03-ab55-278ff0ae92b0)

Regarding ROC curves, although QDA seems not as good as others, it’s not significant. Therefore, we conclude that all the models are quite similar.

AUC of different models appear to be not different significantly from one another.

## Summary
Based on analysis above, we believe that all models act quite good similarly. However,  LDA appears to be quite better than other models. 
