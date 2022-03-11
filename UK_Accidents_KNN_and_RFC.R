library(fastDummies)
library(ggplot2)
library(moments)
library(car)
library(performance)
library(see)
library(patchwork)
library(rpart)
library(rpart.plot)
library(dplyr)
library(class)
library(caret)
library(randomForest)
library(ISLR)
library(class)
library(MASS)

#Reading the dataset file
dfAcc= read.csv("Accidents0515.csv")

#converted accident date column from character to date
dfAcc$Date <- as.Date(dfAcc$Date,format = "%d/%m/%Y")

#Filtered data to get only from 2011 till 2015
dfAcc <- dfAcc[dfAcc$Date >="2014-11-01" & dfAcc$Date <= "2014-12-31",]

#omitting "unallocated" in the urban_rural_area in Accidents dataset and convert 2 level to binary level factors
dfAcc$Urban_or_Rural_Area[dfAcc$Urban_or_Rural_Area=="1"] <- 1
dfAcc$Urban_or_Rural_Area[dfAcc$Urban_or_Rural_Area=="2"] <- 0
dfAcc$Urban_or_Rural_Area[dfAcc$Urban_or_Rural_Area=="3"] <- NA

#adding "Fatal" category into the serious category in Accident_severity- Accidents dataset and convert 2 level to binary level factors
dfAcc$Accident_Severity[dfAcc$Accident_Severity=="1"] <- 1 # Fatal
dfAcc$Accident_Severity[dfAcc$Accident_Severity=="2"] <- 1 # Serious
dfAcc$Accident_Severity[dfAcc$Accident_Severity=="3"] <- 0 # Slight


#Renaming column name from "urban_rural_area" to "AreaType"
colnames(dfAcc)[30] <-"Area_type"

#Removing na values
dfAcc<- na.omit(dfAcc)

#converting accidents required variables from numeric to factors
dfAcc$Police_Force=as.factor(dfAcc$Police_Force)
dfAcc$Accident_Severity=as.factor(dfAcc$Accident_Severity)
dfAcc$Day_of_Week=as.factor(dfAcc$Day_of_Week)
dfAcc$Road_Type=as.factor(dfAcc$Road_Type)
dfAcc$Speed_limit=as.factor(dfAcc$Speed_limit)
dfAcc$Junction_Detail=as.factor(dfAcc$Junction_Detail)
dfAcc$Junction_Control=as.factor(dfAcc$Junction_Control)
dfAcc$Light_Conditions=as.factor(dfAcc$Light_Conditions)
dfAcc$Weather_Conditions=as.factor(dfAcc$Weather_Conditions)
dfAcc$Road_Surface_Conditions=as.factor(dfAcc$Road_Surface_Conditions)
dfAcc$Special_Conditions_at_Site=as.factor(dfAcc$Special_Conditions_at_Site)
dfAcc$Junction_Control=as.factor(dfAcc$Junction_Control)
dfAcc$X2nd_Road_Class =as.factor(dfAcc$X2nd_Road_Class )
dfAcc$Junction_Control=as.factor(dfAcc$Junction_Control)
dfAcc$Area_type=as.factor(dfAcc$Area_type)

#Selecting the subset of the dataframe with relevant independent variables selected
acc.subset <-dfAcc[c("Accident_Severity","Area_type","Number_of_Vehicles"
                     ,"Number_of_Casualties","X1st_Road_Class","Road_Type","Speed_limit",
                     "Junction_Detail","Junction_Control","X2nd_Road_Class","Light_Conditions",
                     "Weather_Conditions","Road_Surface_Conditions","Special_Conditions_at_Site",
                     "Carriageway_Hazards","Did_Police_Officer_Attend_Scene_of_Accident"
                  )]

#Normalize the data
normalize <- function(x){
  print(x)
  return  ((x-min(x))/(max(x)-min(x)))
}

#Total number of columns
totlcol <- ncol(acc.subset)

#Selecting the subset of the dataframe
acc.subset.n <- as.data.frame(acc.subset[,2:totlcol])

#setting seed value
set.seed(123)

# Splitting data into training and testing set of 70 -30 
dat.d <-sample(1:nrow(acc.subset.n),size=nrow(acc.subset.n)*0.7,replace=FALSE)

#Training data creation
train.acc <- acc.subset[dat.d,]

#Training data creation
test.acc <- acc.subset[-dat.d,]

#Creating dataframe for severity feature which is our target
train.acc_labels <- acc.subset[dat.d,1]
test.acc_labels <- acc.subset[-dat.d,1]

##################### K- Nearest Neighbour #####################

# Finding the k value to apply it on the model
i=1
k.optm=1
for (i in 1:21){
  knn.mod <- knn(train= train.acc, test = test.acc, cl= train.acc_labels,k=i)
  k.optm[i] <- 100 * sum(test.acc_labels == knn.mod)/NROW(test.acc_labels)
  k=i
  cat(k,"=",k.optm[i],"\n")
}


#K=9
knn.9 <- knn(train=train.acc,test=test.acc,cl=train.acc_labels,k=9)
knn.9

#Calculate the proportion of correct classification for K=9
acc.9 <- 100 * sum(test.acc_labels == knn.9)/NROW(test.acc_labels)
acc.9

#Confusion Matrix for k=9
confusionMatrix(table(knn.9,test.acc_labels))

#K=15
knn.13 <- knn(train=train.acc,test=test.acc,cl=train.acc_labels,k=13)
knn.13

#Calculate the proportion of correct classification for K=13
acc.13 <- 100 * sum(test.acc_labels == knn.13)/NROW(test.acc_labels)
acc.13

#Confusion Matrix for k=13
confusionMatrix(table(knn.13,test.acc_labels))

#K=21
knn.21 <- knn(train=train.acc,test=test.acc,cl=train.acc_labels,k=21)
knn.21

#Calculate the proportion of correct classification for K=21
acc.21 <- 100 * sum(test.acc_labels == knn.21)/NROW(test.acc_labels)
acc.21

#Confusion Matrix for k=21
confusionMatrix(table(knn.21,test.acc_labels))


######################### Random Forest #############################
rf <- randomForest(Accident_Severity ~.,data=train.acc)
print(rf)
attributes(rf)
summary(rf)
rf$confusion


#Predict and confusion matrix - train data
dtree = rpart(Accident_Severity ~.,data=train.acc,control=rpart.control(10))
dtree$variable.importance

p1 <- predict(rf,train.acc)

head(p1)
head(train.acc$Accident_Severity)
confusionMatrix(p1,test.acc$Accident_Severity)
plot(p1)

#Predict and confusion matrix - test data
p2 <- predict(rf,test.acc)

head(p2)
head(test.acc$Accident_Severity)
confusionMatrix(p2,test.acc$Accident_Severity)
plot(p2)

##Model Tuning 1: 10 folds cross-validation
library(mlbench)
library(caret)
library(e1071)
control2 <- trainControl(method='repeatedcv', number=5, repeats=3)
metric <- "Accuracy"
set.seed(123)
mtry <- 1
tunegrid <- expand.grid(.mtry=mtry)
rf_default2 <- train(Accident_Severity ~., data=train.acc, method='rf', metric='Accuracy', tuneGrid=tunegrid, trControl=control2)
print(rf_default2)

##Model Tuning 2: Random parameter
mtry <- 1
#ntree: Number of trees to grow.
ntree <- 3

control2_2 <- trainControl(method='repeatedcv', number=10, repeats=3,search = 'random')


#Random generate 15 mtry values with tuneLength = 15
set.seed(1)
rf_random2 <- train(Accident_Severity ~.,data=train.acc,method = 'rf',metric = 'Accuracy',tuneLength  = 15, trControl = control2_2)
print(rf_random2)

#Final Results

#Testing the model with the train data
trainPred_Final=predict(rf_default2,train.acc)
head(trainPred_Final)
head(train.acc$Accident_Severity)
confusionMatrix(trainPred_Final,train.acc$Accident_Severity)

#Testing the model with the test data
testPred_Final=predict(rf_default2,test.acc)
head(testPred_Final)
head(test.acc$Accident_Severity)
confusionMatrix(testPred_Final,test.acc$Accident_Severity)

#Plotting final model
plot(rf2)


