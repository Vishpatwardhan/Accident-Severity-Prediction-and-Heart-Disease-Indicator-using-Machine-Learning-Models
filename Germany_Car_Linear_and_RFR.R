library(fastDummies)
library(ggplot2)
library(moments)
library(car)
library(performance)
library(see)
library(patchwork)
library(stringr)
library(caret)
library(randomForest)
library(ranger)
library(tidyverse)

#Reading the dataset file
df= read.csv("autoscout24-germany-dataset.csv")
names(df) <-gsub("\\.","",names(df))


#Removing the blank values
df$gear[df$gear==""] <- NA
df$fuel[df$fuel == "(Fuel)"] <- NA

#Replacing the special characters in the variable names
df$gear <- str_replace_all(df$gear, '-', '')
df$fuel <- str_replace_all(df$fuel, '/', '')
df$fuel <- str_replace_all(df$fuel, '--', '')
df$fuel <- str_replace_all(df$fuel, ' ', '')
df$fuel <- str_replace_all(df$fuel, '(', '')
df$fuel <- str_replace_all(df$fuel, ')', '')
df$fuel <- str_replace_all(df$fuel, '(Fuel)', 'Fuel')

#Converting offer type to factor
df$offerType<- as.factor(df$offerType)

df$gear<- as.factor(df$gear)
df=dummy_cols(df,select_columns="gear")

df$fuel<- as.factor(df$fuel)
df=dummy_cols(df,select_columns="fuel")

df$handDrive<- as.factor(df$handDrive)
df=dummy_cols(df,select_columns="handDrive")

#Eliminating na values in the dataframe
df<- na.omit(df)


par(mfrow=c(1,2))

hist(df$price,main="Histogram for Price",xlab="Price (Before transformation)",ylab="Frequency",col="#9fdf9f")
hist(df$hp,main="Histogram for Horsepower",xlab="Horsepower (Before transformation)",ylab="Frequency",col="#9fdf9f")
hist(df$mileage,main="Histogram for Mileage",xlab="Mileage (Before transformation)",ylab="Frequency",col="#9fdf9f")

df_corr <- df[,c("price","hp","gear_Automatic","gear_Manual","gear_Semiautomatic","fuel_CNG","fuel_ElectricDiesel","fuel_ElectricGasoline","fuel_Ethanol","fuel_Gasoline",
                 "fuel_Hydrogen","fuel_LPG","fuel_Others","mileage","handDrive_LHD","handDrive_RHD")]
df_corr

#Correlation matrix
library(corrplot)
correlations <- cor(df_corr)
corrplot(correlations, method="circle")

boxplot(df[,c("price","mileage","hp")],main="Histogram for Price,Mileage, and Horsepower",ylab="Frequency",col="#ff8080")

#Transformation to treat the outliers
newprice <- 19490 + 1.5 * IQR(df$price)
newprice <- df$price[df$price < newprice]

newhp <- 150 + 1.5 * IQR(df$hp)
newhp <- df$hp[df$hp < newhp]

newmileage <- 105000 + 1.5 * IQR(df$mileage)
newmileage <- df$mileage[df$mileage < newmileage]

#remove remaining outliers
df <- subset(df,df$price <= 18990 & df$mileage <=114387.6 & df$hp <=246)

#Final Transformation
df$price=sqrt(df$price)
df$hp=sqrt(df$hp)

hist(df$price,main="Histogram for Price",xlab="Price (After sqrt transformation)",ylab="Frequency",col="#9fdf9f")
hist(df$hp,main="Histogram for Horsepower",xlab="Horsepower (After sqrt transformation)",ylab="Frequency",col="#9fdf9f")
hist(df$mileage,main="Histogram for Mileage",xlab="Mileage",ylab="Frequency",col="#9fdf9f")


#to achieve reproducible model; setting the random seed number
set.seed(123)

#create partition of the data
TrainingIndex <- createDataPartition(df$price,p=0.8,list = FALSE)
train <- df[TrainingIndex,]
test <- df[-TrainingIndex,]

str(train)
str(test)

##################### Multiple Linear Regression #######################

#Basic model using all the relevant independent variables
mod1 <- lm(price ~ hp + gear_Automatic + gear_Manual + gear_Semiautomatic +fuel_CNG+fuel_Diesel+
             fuel_Electric+fuel_ElectricDiesel+fuel_ElectricGasoline+fuel_Ethanol+fuel_Gasoline
           +fuel_Hydrogen+fuel_LPG+fuel_Others+mileage+handDrive_LHD+handDrive_RHD,data=df)
summary(mod1)

#Final model using all the significant independent variables
mod2 <- lm(price ~ hp +fuel_CNG+fuel_Diesel+fuel_ElectricGasoline + mileage, data=test)
summary(mod2)

#Adding the final model in mymodel
mymodel <-mod2

#par(mfrow=c(2,2))
plot(mymodel,col="#21BDA9")

#VIF
vif(mymodel)

#Durbin-Watson test
durbinWatsonTest(mymodel)

#ncvTest
ncvTest(mymodel)

predictions <- mymodel %>% predict(test)

# Model performance
# (a) Prediction error, RMSE
RMSE(predictions, test$price)


#Plotting for train vs test model
plot(pred_train,train$price)

#Plotting the model with respect to test dataset.
plot(predict(mymodel,newdata=test),test$price,col="#00e6e6")
abline(line(pred,test$price), col = "#006666", lwd = 3)

#Final model plotting
plot(mymodel,col="#21BDA9")

############# Random Forest ###################################

#Randome Forest Model
system.time(rf_fit <- train(price ~ hp +fuel_CNG+fuel_Diesel+fuel_ElectricGasoline + mileage, data=train,method="rf", importance =TRUE, trControl=trainControl(method="cv",number=10)))
rf_fit

#evaluate variable importance 
varImp(rf_fit)

#RMSE evaluation
rsme_rf<-sqrt(mean((test$price-predict(rf_fit,test))^2))
rsme_rf

#Plotting Actual vs Predicted for final random forest model
plot(predict(rf_fit,test),test$price,col="#00e6e6")
abline(line(predict(rf_fit,test),test$price), col = "#006666", lwd = 3)




