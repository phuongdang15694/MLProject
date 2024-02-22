 Mortality Rate Modeling Project

Nowadays, there are many ways to determine the overall condition of a country, one of those is to make a good observation of the vital statistics, such as birth and death rates, and those numbers 
may reveal a lot of underlying issues of a society, especially in the long run. In this project, we will
apply machine learning methods in forecasting the mortality rate in Canada through demographic
and socioeconomic characteristics.

THE DATASET

The data set includes four predictors - that are, people with low-income percentage, a proportion
of 65-year-olds and over population, and unemployment rate, one response variable - mortality
rate, with 100 observations. Information was extracted from Statistics Canada along with their
metadata. Data for all variables is in only available for 10 years, from 2010 to 2019, just before the
onset of Covid-19.

Region (named in the dataset) defined as official provinces of Canada (not including territories): Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, and Saskatchewan. Along with the province, the annual population estimates of each province from 2010 to 2019 were added as well. This data is from table 17-10-0005-01 (formerly CANSIM 051-0001) and released in 2022. 

Low-income percentage (Income) is a continuous variable, which is measured by taking estimates of the population with low income divided by the total population and multiplied by 100. Values were extracted from the percentage of Low-income cut-offs after tax (LICO-AT) 1992 base, in table: 11-10-0135-01 (formerly CANSIM 206-0041) of Centre for Income and Socioeconomic Well-being Statistics, Statistics Canada. LICO-AT 1992 base is defined as: “The low-income cut-offs after tax (LICO-AT) are income thresholds below which a family will likely devote a larger share of its after-tax income on the necessities of food, shelter, and clothing than the average family. The approach is essentially to estimate an income threshold at which families are expected to spend 20 percentage points more than the average family on food, shelter, and clothing, based on the 1992 Family Expenditures Survey. LICOs are calculated in this manner for seven family sizes and five community sizes.” (Low-income cut-offs (LICOs) before and after tax by community size and family size, in current dollars, 2023). 

Senior percentage (Age) is a continuous variable, measured by taking the estimates of people 65 years old and over devided by total population, and then multiplied by 100. Data were taken from table: 17-10-0005-01 (formerly CANSIM 051-0001). 

Unemployment percentage (Unemployment), which is a continuous variable, is the annual unemployed population divided by the total population, and multiplied by 100. Data is from the Labour Force Survey, Statistics Canada. 

Regarding the response variable, the annual mortality percentage is from Birth and Death Databases and the Centre for Demography, Statistics Canada. 

DATA PREPARATION AND EXPLORATION

Before applying machine learning algorithms, we initially explore the quality and correlations in the dataset. Overall, there are no missing values in all of the database and the quality of data is ranked D, meaning “Acceptable”, and above. We then plot trends of variables over 10 years by provinces. 

We can take a brief look at the relationship among variables. As seen in Figure 1 below, we found Death and Age have a moderately strong positive correlation (corr = 0.785), while Death is weakly correlated with Unemployment (corr = 0.493). Income is negatively correlated with Death (corr = -0.312), yet this plot shows a weaker correlation than might be expected. We can also notice there is a weak correlation between Age and Unemployment (corr = 0.432) which should be careful as collinearity can be an issue while building the model. Finally,  as region is a categorical variable, we decide using dummy coding. It's a way to turn a categorical variable into a series of binary variables (variables that can have a value of zero or just one value). 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/7ec2c864-3ae6-49e3-84ed-5f75e9369976)

Figure 1.  Plotting the Mortality dataset with ggpairs

RESULT

Since the response variable is continuous and most of the predictors are continuous variables, we use linear regression to build the model. We initially applied linear regression in our original dataset. This model gave an  R-Squared (R2) score of 0.9647, RSE of 0.2339, and coefficients as follows: 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/c4261560-33a9-4a74-9da0-e7501cfec0e7)

   Figure 2. Performance of linear model from the original dataset
   
The summary of the model demonstrates comprehensive information about each regressor and how they contribute to the response variable. We can notice that the algorithm could not offer us the coefficient for the last predictor, and the contributions of half of the predictors are not significant. We can presume that there exists collinearity among the independent variables in all of the data. This leads us to the next step, which is to exam the use of PCA . 

After scaling and centering the dataset, we performed PCA to reduce dimensions and mitigate the impact of multicollinearity. The Scree Plot in Figure 3 illustrates how much each PC accounted for percentage variance. As can be seen, there are huge gaps between PC1 - PC2, PC9 - PC10. Thus, we should consider at the table of PCA summary below for the Cumulative Proportion, which helps us determine  that we should retain the top 9 principal components. In the end, we decided to develop a new model based on new variables created by PCA. 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/659867a4-9ff8-4f61-bfbd-ea5590d44c39)


Figure 3. Scree Plot for PCA 

The linear model has shown a noticeable improvement post PCA as compared to the former one Firstly, the model is now capable of measuring sufficient coefficients for the predictors, which is a good sign of successful solution to the problem of collinearity. Furthermore, all the predictors have made significant contributions to the response variable. Although Residual Standard Error (RSE) has increased and R2 has decreased, these changes are all explicable as we have only selected the important principal components. 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/add667bc-981f-49d2-add5-e71bb54612a3)

 Figure 4. Performance of linear model from the dataset after perorming PCA

 To ensure the absence of redundant regressor, we utilize an additional method with a less subjective approach to feature selection, which allow another algorithm to evaluate relevant features. We use filter method to compare each predictor to the response variable. As per Figure 5w, which ranks the degree of information that each independent variable can contribute to the model. As a result, we decide to retain all nine predictors. 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/d2ab0069-86b9-4ea1-87e6-2e3d1de96673)

  Figure 5. Feature selection for the new linear model

Following feature selection, it is necessary to verify 4 assumptions of linear regression: Linearity, Equal Variance, Independence, and Normality. Given that PCA was used to eliminate collinearity,  it is acceptable to skip the Independence test. 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/73a579d6-06ca-43be-930d-9ba4e2045b1a)

   Figure 6. Fitted vs Residuals Plot

Fitted versus Residuals Plot in Figure 6 is used for checking both the Linearity and Equal Variance assumptions. The Linearity test reveals whether the relationship between dependent and independent variables is linear. We observe that at any fitted value, the mean of the residuals is roughly 0. For this reason, we can conclude that the linearity assumption is not violated, and it indicates we have a linear relationship between predictors and response variables. On the other hand, there is heteroscedasticity in this model since the spread of the residuals is not consistent across all  variables. Thus this violation of the Equal Variance assumption indicates the presence of heteroscedasticity. 

In order to check for Normality assumption, we determine whether residuals are normally distributed by performing a Q-Q Plot and a Histogram Plot. In Figure 7, the points of Q-Q plot do not closely follow the line, suggesting that the errors may not follow a normal distribution. The histogram appears not to display a bell-shaped curve, with its skewness and kurtosis is -0.2440806  and 3.33, respectively, inferring a slightly left-skewed and fat-tailed graph. These observations lead us to suspect that a violation of normality. Since graph visualization is insufficient to draw a conclusion,the Shapiro-Wilk test should be used with hypothesis H0: The distribution follows a normal distribution. This normality test yielded a p-value of 0.2592, which is greater than alpha of 0.05, we cannot reject the hypothesis that the distribution follows a normal distribution. 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/d6f27bdd-49de-4794-9b32-d8ce82d82016)
![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/5d4ceb7a-debe-4073-83cb-35e932f5adfd)

  Figure 7. Q-Q Plot and Histogram of Residuals

  We now predict mortality rate in testing set using the newly constructed model and then compare results to the observed rates. The association between the predicted and observed values is subsequently validated by implementing a linear regression. Both Root Mean Squared Error (RMSE) and R2 are used to examine the model. R2 offers information about the goodness of fit of a model. It is the rate of the sum of explained variation to the total variation: 

  ![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/43f67758-b2a8-49d8-9068-86777c4fd7ed)

As a proportion, the range of R2 is from 0 to 1. In the case where R2 = 1, every data point lies perfectly on the regression line, meaning the predicted values the observations are identical. On the other hand, an R2 value of 0 implies that the our model to predict mortality rate fails. In this case,  the R2 is 0.9796, which is pretty close to 1, indicating that 97.96% of the observed values can be accounted for by the predicted values.

In contrast, RMSE evaluates the standard deviation of residuals. RMSE is a good measure of how accurately the model predicts the response, and is the most important criterion for fit if the main purpose of the model is prediction. Lower value of RMSE denotes a better fit. The formula of RMSE is as follows:

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/1946c368-d42f-435c-9646-079a3c5f4342)

As can be seen, there is no rule of thumb regarding good or bad RMSE because it depends on the absolute value of dependent variable, which is the observed mortality rate. Our mean, min and max of mortality rate are approximately 8, 5 and 10, respectively, while RMSE is 0.2, roughly equal to 2.5% of the response value. It means that there is only 2.5% of discrepancy between the predicted and observed values. Figure 8 depicts the data points are distributed closely around the regression line, indicating a degree of variance in observed values in charge in predicted ones. In general, the graph suggest a robust positive linear regression between those two variables. 

![image](https://github.com/phuongdang15694/Machine-Learning-Project/assets/103254136/bff502c1-6c4b-405d-8248-563168efd869)

 Figure 8. Linear model  between predicted values and observed values

 Outcomes and Limitations
Some limitations remain with regard to this model. To begin with, it  has low interpretability as despite Principal components are linear combinations of the features from the original data, they are not as easy to interpret.  While the model effectively predicts the mortality rate using the provided dataset, it is not easily determine the individual contribution of each regressor to the mortality. In lieu of PCA, we can employ backward or stepwise selection using Akaike information criterion (AIC) or the Bayesian information criterion (BIC). Secondly, there is still an issue with the condition of heteroscedasticity, which requires a transformation of response variables using Box-Cox method, which aims to find a more exact form for the relation. Finally, the potential concern about overfitting may happen due to the small size of the dataset. In spite of all these constraints, we have constructed a linear model to predict the mortality rate of Canada with give predictors. 
