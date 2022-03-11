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
library(caTools)
library(ROCR)

#Reading the dataset file
data= read.csv("heart_disease_health_indicators_BRFSS2015.csv")
names(data) <-gsub("\\.","",names(data))

#converting required variables from numeric data to factors
data$HeartDiseaseorAttack=as.factor(data$HeartDiseaseorAttack)
data$HighBP=as.factor(data$HighBP)
data$HighChol=as.factor(data$HighChol)
data$CholCheck=as.factor(data$CholCheck)
data$Smoker=as.factor(data$Smoker)
data$Stroke=as.factor(data$Stroke)
data$Diabetes=as.factor(data$Diabetes)
data$PhysActivity=as.factor(data$PhysActivity)
data$Fruits=as.factor(data$Fruits)
data$Veggies=as.factor(data$Veggies)
data$HvyAlcoholConsump=as.factor(data$HvyAlcoholConsump)
data$AnyHealthcare=as.factor(data$AnyHealthcare)
data$NoDocbcCost =as.factor(data$NoDocbcCost )
data$GenHlth=as.factor(data$GenHlth)
data$MentHlth=as.factor(data$MentHlth)
data$PhysHlth=as.factor(data$PhysHlth)
data$DiffWalk=as.factor(data$DiffWalk)
data$Sex=as.factor(data$Sex )
data$Age=as.factor(data$Age)
data$Education=as.factor(data$Education)
data$Income=as.factor(data$Income)

#Removing na values
data<- na.omit(data)

#Setting the seed
set.seed(123)

#splitting the data set into 80-20 ratio for training and testing respectively
split <- sample.split(data,SplitRatio = 0.8)

#Training data creation
training <- subset(data,split == "TRUE")

#Testing data creation
testing <- subset(data,split == "FALSE")



########################## Logistic Regression Model ##########################
model <- glm(HeartDiseaseorAttack ~ HighBP+HighChol+CholCheck+BMI+Smoker+Stroke+Diabetes+PhysActivity+Fruits+Veggies
             +HvyAlcoholConsump+AnyHealthcare+NoDocbcCost,training, family="binomial")

#Summary of the model
summary(model)

#response from the prediction of model vs test data
res <- predict(model,testing,type="response")
#res

#displaying table of actual vs predicted value for analysization
table(ActualValue=testing$HeartDiseaseorAttack, PredictedValue=res>0.5)

#Initialization of ROC curve plotting
ROCRPred <- prediction(res,testing$HeartDiseaseorAttack)
ROCRPref <- performance(ROCRPred,"tpr","fpr")

#Plotting ROC Curve
plot(ROCRPref,colorize=TRUE, print.cutoffs.at=seq(0.1,by=0.1))

#Calculating the accuracy of random response values
table(ActualValue=testing$HeartDiseaseorAttack, PredictedValue=res>0.2)
print(47857+1922)/(47857+4512+3364+1922)

table(ActualValue=testing$HeartDiseaseorAttack, PredictedValue=res>0.4)
print(51811+540)/(51811+558+4746+540)

table(ActualValue=testing$HeartDiseaseorAttack, PredictedValue=res>0.1)
print(37713+3734)/(37713+3734+14656+1552)

#Summary of final model
summary(model)


################## Decision Tree ##########################

library(party)
names(train_pd)

par(mfrow=c(1,1))

#Basic decision tree model
tree <- rpart(HeartDiseaseorAttack ~ HighBP+HighChol+CholCheck+BMI+Smoker+Stroke+Diabetes+PhysActivity+Fruits+Veggies
              +HvyAlcoholConsump+AnyHealthcare+NoDocbcCost,data=training,method="class",control=rpart.control(minsplit =100,minbucket=100, cp=0)) #,control=rpart.control(minsplit =100,minbucket=500, cp=0)

plot(tree)
text(tree)
summary(tree)
printcp(tree)
plotcp(tree)
rpart.plot(tree,cex=0.5,extra=0)

#prediction model on testing data
predict_pd<-predict(tree,testing,type = "class")
#predict_pd


#confusion matrix of predicted testing model vs actual data
confusionMatrix(predict_pd,testing$HeartDiseaseorAttack)


#cross validation and pruning
tree2<-rpart(HeartDiseaseorAttack ~ HighBP+HighChol+CholCheck+BMI+Smoker+Stroke+Diabetes+PhysActivity+Fruits+Veggies
                +HvyAlcoholConsump+AnyHealthcare+NoDocbcCost,data=training,method="class",
                parms=list(split="Information"),control = rpart.control(cp=0.0002,minsplit = 500,minbucket = 100,maxdepth = 15,xval=10))

printcp(tree2)
rpart.plot(tree2,cex = 0.5)
plotcp(tree2,col="blue")

#Complexity parameter value = 0.00053743

#Pruning the tree2
pd_prune<-prune(tree2,0.00053743)
rpart.plot(pd_prune,cex = 0.5)

#Generating decision tree by applying the complexity parameter as 0.00053743
pd_tree3<-rpart(HeartDiseaseorAttack ~ HighBP+HighChol+CholCheck+BMI+Smoker+Stroke+Diabetes+PhysActivity+Fruits+Veggies
                +HvyAlcoholConsump+AnyHealthcare+NoDocbcCost,data=training,method="class",
                parms=list(split="Information"),control = rpart.control(cp=0.00053743,minsplit = 500,minbucket = 100,maxdepth = 15))

#plotting the Decision Tree - (tree3)
rpart.plot(pd_tree3,cex = 0.5)

#Initializing the Final Prediction of the optimized decision tree model
predict_final<-predict(pd_tree3,testing,type = "class")
#predict_final

#Final Confusion Matrix 
confusionMatrix(predict_final,testing$HeartDiseaseorAttack)



