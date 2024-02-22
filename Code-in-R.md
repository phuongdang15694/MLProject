--LIBRARY--

library(mlr)	

library(tidyverse)	

library(dplyr)

library(ggplot2)

library(fastDummies)

library(psych)

library(factoextra)

library(GGally)

library(moments)

library(caret)

library(Metrics)

--DATASET OVERVIEW--

project_ln_full = read.csv("D:\\OC\\Semester 2\\MLI 400\\Project\\Data\\Project_lr.csv")

project_ln_full_Tib = as.tibble(project_ln_full[-1])

summary(project_ln_full_Tib)

project_plot=read.csv("D:\\OC\\Semester 2\\MLI 400\\Project\\Data\\Project_lr _ Acronym province.csv")

ggpairs(project_plot,upper = list(continuous="cor",combo = "facetdensity"),lower = list(continuous=wrap("smooth", alpha = 0.3, size=0.1),combo = "box_no_facet"))
 +theme(axis.text.x = element_text(angle = 90, hjust = 1))
 
pairs.panels(project_plot,smooth=FALSE,lm=TRUE)

--DATA MANIPULATION--

project_ln_full_Tib_dummy = dummy_cols(project_ln_full_Tib,select_columns="Region")

project_ln_full_Tib_dummy_noregion=project_ln_full_Tib_dummy[,-1]

project_full= project_ln_full_Tib_dummy_noregion[sample(1:nrow(project_ln_full_Tib_dummy)),]

project_train=project_full[1:80,]

project_test = project_full[81:100,]

colnames(project_train)=c("Income","Seniors","Unemployment","Mortality","AB","BC","MB","NB","NL","NS","ON","PE","QC","SK")

colnames(project_test)=c("Income","Seniors","Unemployment","Mortality","AB","BC","MB","NB","NL","NS","ON","PE","QC","SK")

--TRAINING--

 -LINEAR REGRESSION-

project_train.lm= lm(Mortality~.,data=project_train)

-PRINCIPLE COMPONENT ANALYSIS-

pca = select (project_train, -Mortality,) %>% 
  prcomp(center=TRUE, scale=TRUE)
  
pca

pcaDat <- get_pca(pca)

fviz_pca_biplot(pca, label = "var")+ theme_bw()

fviz_pca_var(pca)

fviz_screeplot(pca, addlabels = TRUE, choice = "eigenvalue")

fviz_screeplot(pca, addlabels = TRUE, choice = "variance")

projectPca = project_train%>% 
  mutate(PC1 = pca$x[, 1], PC2 = pca$x[, 2],PC3 = pca$x[, 3],PC4 = pca$x[, 4],
         PC5 = pca$x[, 5],PC6 = pca$x[, 6],PC7 = pca$x[, 7],PC8 = pca$x[, 8],
         PC9 = pca$x[, 9])
project_train_Pca = as.tibble(projectPca[,c("Mortality","PC1","PC2","PC3","PC4"
                                            ,"PC5","PC6","PC7","PC8","PC9")])
                                            
ggplot(project_train_Pca, aes(PC1, PC2,  col = Mortality)) +
  geom_point() +
  theme_bw()

project_train_Pca

project_train_Pca.lm=lm(Mortality~PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9,
                        data=project_train_Pca)

summary(project_train_Pca.lm)

-FEATURE SELECTION-

projectTask = makeRegrTask(data=project_train_Pca,target = "Mortality")

filterVals <- generateFilterValuesData(projectTask,
                                       method = "linear.correlation")
                                       
filterVals$data

plotFilterValues(filterVals) + theme_bw()

-NORMALITY CHECK-

hist(resid(project_train_Pca.lm),
     xlab = "Residual",
     main = "Histogram of Residuals",
     col = "grey",
     border = "dodgerblue",
     breaks = 15)
     
skewness(resid(project_train_Pca.lm))

kurtosis(resid(project_train_Pca.lm))

par(mfrow = c(1, 2))

project_train_Pca.lm$residuals

project_train_Pca.lm$fitted.values

plot(fitted(project_train_Pca.lm), resid(project_train_Pca.lm), col = "grey", pch = 20, xlab = "Fitted", ylab = "Residuals", main = "Fitted vs Residuals Plot")
abline(h = 0, col = "darkorange", lwd = 2)

qqnorm(resid(project_train_Pca.lm), main="QQ Plot", col="red")

qqline(resid(project_train_Pca.lm),col="blue",lwd=2)

shapiro.test(resid(project_train_Pca.lm))

-CROSS VALIDATION-

CV= trainControl(method="repeatedcv",number=8,repeats = 10)

CV_train=train(Mortality~PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9,data=project_train_Pca,method="lm",trControl=CV)

CV_train

--TESTING--

project_test_pca = predict(pca,project_test) 

project_test_pca_df=as.data.frame(project_test_pca)

predict_trial=predict(project_train_Pca.lm,newdata =project_test_pca_df)

predict_trial


data.frame(R2 = R2(predict_trial, project_test$Mortality),
          RMSE = RMSE(predict_trial, project_test$Mortality),
          MAE = MAE(predict_trial, project_test$Mortality))
            
project_test_predict_full=mutate(project_test,predict_trial)

colnames(project_test_predict_full)[15]="Mortality_predicted"

observed_predict.lm = lm(Mortality~Mortality_predicted,project_test_predict_full)

summary(observed_predict.lm)

plot(project_test_predict_full$Mortality,project_test_predict_full$Mortality_predicted, ylab = "Observed",xlab = "Predicted")
+abline(observed_predict.lm,col="orange", lwd=2)
