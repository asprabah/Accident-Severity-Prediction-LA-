##################################

#Aravindh Siddharth

#Accident Severity Prediction LA

##################################

library(rpart)
library(gbm)
library(ada)
library(randomForest)
library(caret)
library(car)
library(ggmap)
library(ggplot2)


#setting working directory
setwd('C:/Users/Flynn/Desktop/Data analytics proj')

getwd()

#reading fulldata containing initial dataset with 2.25 million records for whole United States of America
fulldata <- read.csv('accidents12.csv')

#subsetting for LosAngeles only
laonly <- subset(fulldata, fulldata$City == 'Los Angeles')

#converting severity to two levels
laonly$Severity[laonly$Severity < 3] <- 1
laonly$Severity[laonly$Severity == 3] <- 2
laonly$Severity[laonly$Severity > 3] <- 2

laonly$Severity <- as.factor(laonly$Severity)

str(laonly)

summary(laonly)

#removing redundant variables
laonlyreqvar <- laonly[,c(4,7,8,15,16,17,24,25,26,27,28,30,31,32,34,37,42,44,46)]

finaldataset <- laonlyreqvar[,-c(5,6,8,12,13,15,17)]

str(finaldataset)

#lat and long used to plot in map
finaldatawithlatandlong <- finaldataset

#removing lat and long to create a modeling dataset
finalmodelingdataset <- finaldataset[, -c(2,3)]

str(finalmodelingdataset)

#checking and removing na values
sum(is.na(finalmodelingdataset))
finalmodelingdataset <- na.omit(finalmodelingdataset)
str(finalmodelingdataset)

#converting visibility to factor
finalmodelingdataset$Visibility.mi. <- as.factor(finalmodelingdataset$Visibility.mi.)

#converting weather conditioin to 4 level factor
#install.packages("car")
library(car)
finalmodelingdataset$Weather_Condition <- recode(finalmodelingdataset$Weather_Condition,"c('Drizzle','Heavy Rain','Light Drizzle','Light Rain','Light Thunderstorms and Rain','Rain','Thunderstorm')='rain';c('Mostly Cloudy','Overcast','Partly Cloudy','Scattered Clouds')='cloudy';c('Smoke','Fog','Haze','Mist','Patches of Fog','Shallow Fog')='fog'")
finalmodelingdataset <- finalmodelingdataset[!finalmodelingdataset$Weather_Condition == "",]
str(finalmodelingdataset)

#saving the final dataset to final.csv
write.csv(finalmodelingdataset, file = "final.csv")

######### Project starts from here ##########

laaccident <- read.csv('final.csv')
laaccident <- laaccident[,-c(1)] #used to remove the first index column that has been created while saving the new csv file
str(laaccident)

laaccident$Severity <- as.factor(laaccident$Severity)
laaccident$Visibility.mi. <- as.factor(laaccident$Visibility.mi.)

##########################################Exploratory Data Analysis#########################################################
dev.off()

#plotting response variable
plot(laaccident$Severity, ylim = c(0, 30000), main = "Response variable", col = 'pink', names = c('low severity','high severity'))

#using ggmap to plot the datapoints on Los Angeles map 

incidents <- finaldatawithlatandlong
#install.packages("ggmap")
library(ggmap)
ggmap::register_google(key = "AIzaSyCK_MlkB3zLV8Yz-T-8yOIaNqVUNVpn_do")

#taking Los angeles map from googlemaps and plotting all datapoints in the map
p <- ggmap(get_googlemap(maptype="terrain",zoom=11,center = c(lon = -118.28904, lat = 34.078926)))
p + geom_point(aes(x =Start_Lng , y =Start_Lat ),colour = 'red', incidents, alpha=0.25, size = 0.5)

i2lsev <-subset(incidents,incidents$Severity=='1') #subsetting only low severity
i2hsev<-subset(incidents,incidents$Severity=='2')  #subsetting only high severity

#distinguishing high severity as #red and low severity as #yellow
p + geom_point(aes(x =Start_Lng , y =Start_Lat ),colour = 'yellow', i2lsev, alpha=0.25, size = 0.5) + 
  geom_point(aes(x =Start_Lng , y =Start_Lat ),colour = 'red', i2hsev, alpha=0.25, size = 0.5) 

#plotting all predictors

par(mfrow = c(3,3))

hist(laaccident$Temperature.F., main = 'Distribution of temperature',xlab = 'Temperature', col = 'skyblue')

hist(laaccident$Humidity..., main = 'Distribution of humidity', xlab = 'Humidity', col = 'skyblue')

hist(laaccident$Pressure.in., main = 'Distribution of pressure', xlab = 'Pressure', col = 'skyblue')

plot(laaccident$Side, ylim = c(0,50000), main = 'Side', xlab = '', col = 'skyblue')

plot(laaccident$Sunrise_Sunset, main = 'Time of the day', col = 'skyblue', ylim = c(0,35000))

plot(laaccident$Visibility.mi., ylim = c(0,45000), main = 'Visibility', col = 'skyblue')

plot((laaccident$Weather_Condition), ylim = c(0,35000) ,main = 'Weather condition',  col = 'skyblue')

plot(laaccident$Junction, ylim = c(0,50000), col = 'skyblue', main = 'Junction' )

plot(laaccident$Traffic_Signal, ylim = c(0,50000), col = 'skyblue', main = 'traffic signal')

#checking for outliers for the continuous variables using boxplot
par(mfrow = c(1,3))

boxplot(laaccident$Temperature.F., main = 'Boxplot of Temperature', xlab = 'Temperature')
boxplot(laaccident$Humidity..., main = 'Boxplot of Humidity', xlab = 'Humidity')
boxplot(laaccident$Pressure.in., main = 'Boxplot of Pressure', xlab = 'Pressure')

Outlierspressure = data.frame(boxplot(laaccident$Pressure.in., plot=F)$out)
Outlierstemp = data.frame(boxplot(laaccident$Temperature.F., plot=F)$out)
Outliershumid = data.frame(boxplot(laaccident$Humidity..., plot=F)$out)

nrow(Outlierspressure)
nrow(Outlierstemp)
nrow(Outliershumid)

dev.off()

############################################## Fitting Models #####################################################################################  

#Randomized holdout

set.seed(15)
numholdout = 10
percentholdout = 0.2
nmodel = 6

predictionaccuracy <- matrix(data= NA, ncol = nmodel, nrow = numholdout)
trainingaccuracy <- matrix(data= NA, ncol = nmodel, nrow = numholdout)
colnames(predictionaccuracy) <- c("Logistic regression", "Cart using rpart", "Randomforest", "Gbm boost", "Ada boost", "Null model")
colnames(trainingaccuracy) <- c("Logistic regression", "Cart using rpart", "Randomforest", "Gbm boost", "Ada boost", "Null model")


randomstring <- function(percent,length) {
  s <- c()
  for (j in 1:length) {
    if(runif(1) <= percent) {
      s[j] <- 1
    }
    else {
      s[j] <- 0
    }
  }  
  s
}

####### used to get the final model to be used in for loop ######## 
##############################################
trainindex <- sample(x = 1:nrow(laaccident), size = 0.8*(nrow(laaccident)))
train.data <- laaccident[trainindex,]
test.data <- laaccident[-trainindex,]
##############################################
library(caret) #for confusion matrix function

#logistic regression
logistic <- glm(Severity ~  ., data = train.data, family = binomial())
logisticpred <- predict(logistic, newdata = test.data, type = 'response' )

logisticpred <- ifelse(logisticpred > 0.5, "2","1")

summary(logistic) #selecting only significant predictors from summary(logistic)
confusionMatrix(as.factor(logisticpred) , test.data$Severity)

#logistic in for loop
set.seed(13)
attach(laaccident)
for (i in 1:numholdout) {
  s <- randomstring(percentholdout, nrow(laaccident))
  
  tmp.data <- cbind(laaccident,s)
  tmp.response <- (cbind(laaccident$Severity,s))
  holdout <- subset(tmp.data, s==1)[,1:length(laaccident)]
  holdout.response <- subset(tmp.response, s==1)[,1]
  train <- subset(tmp.data, s==0)[,1:length(laaccident)]
  
  sizeholdout <- dim(holdout)[1]
  sizetrain <- dim(train)[1]
  
  #final model after removing insignificant terms
  lm.a <- glm(Severity ~  Side+Humidity...+Pressure.in.+Weather_Condition+Junction+Traffic_Signal+Sunrise_Sunset, data = train, family = binomial())
  lm.a.pred <- predict(lm.a, newdata = holdout, type = 'response' )
  
  lm.a.pred <- ifelse(lm.a.pred > 0.5, "2","1")
  
  lm.train.pred <- predict(lm.a, newdata = train, type = 'response')
  lm.train.pred <- ifelse(lm.train.pred > 0.5, "2","1")
  
  predictionaccuracy[i,1] <- sum(diag(table(lm.a.pred, holdout.response)))/sum(table(lm.a.pred, holdout.response))
  trainingaccuracy[i,1] <- sum(diag(table(lm.train.pred, train$Severity)))/sum(table(lm.train.pred, train$Severity))
  
}
#######################
#rpart
library(rpart)
cart <- rpart(Severity ~ ., train.data, method = "class")
cart.predict <- predict(cart, newdata = test.data, type = 'class')
plot(cart)
text(cart)
confusionMatrix(cart.predict, test.data$Severity)

#rpart in for loop
library(rpart)
set.seed(17)
attach(laaccident)
for (i in 1:numholdout) {
  s <- randomstring(percentholdout, nrow(laaccident))
  
  tmp.data <- cbind(laaccident,s)
  tmp.response <- (cbind(laaccident$Severity,s))
  holdout <- subset(tmp.data, s==1)[,1:length(laaccident)]
  holdout.response <- subset(tmp.response, s==1)[,1]
  train <- subset(tmp.data, s==0)[,1:length(laaccident)]
  
  sizeholdout <- dim(holdout)[1]
  sizetrain <- dim(train)[1]
  
  cartmodel1 <- rpart(Severity ~ ., train, method = "class")
  cart.predict <- predict(cartmodel1, newdata = holdout, type = 'class')
  
  cart.train.pred <- predict(cartmodel1, newdata = train, type = 'class')
  
  predictionaccuracy[i,2] <- sum(diag(table(cart.predict, holdout.response)))/sum(table(cart.predict, holdout.response))
  trainingaccuracy[i,2] <- sum(diag(table(cart.train.pred, train$Severity)))/sum(table(cart.train.pred, train$Severity))
  
}
#######################

#randomforest
library(randomForest)
set.seed(80)
rfmodel <-  randomForest(Severity ~ ., train.data, importance = T )
plot(rfmodel)

rferrorrate <- data.frame(rfmodel$err.rate)
#finding the tree size for minimum error
mintreerf <- which.min(rferrorrate$OOB)
mintreerf #given the optimal tree size

#new rf model with optimal tree size
set.seed(5)
rfmodel1 <- randomForest(Severity ~ ., train.data, ntree = mintreerf, importance = T)
print(rfmodel1)
plot(rfmodel1)

formtry <- c()
for(i in 1:9) {
  temporaryrf <- randomForest(Severity ~., train.data,importance = T, mtry = i, ntree = mintreerf )
  formtry[i] <- temporaryrf$err.rate[mintreerf]
}
formtry #from this we can see the optimal number of predictors

#we will use this optpred and mintreerf in random holdout
optimalmtry <- which.min(formtry)
optimalmtry

finalrfmodel <- randomForest(Severity ~ ., train.data, ntree = mintreerf, mtry = optimalmtry, importance = T  )
plot(finalrfmodel)

rfpredicted <- predict(finalrfmodel, test.data)

confusionMatrix(rfpredicted, test.data$Severity)

#randomforest in forloop
library(randomForest)
set.seed(10)
attach(laaccident)
for (i in 1:numholdout) {
  s <- randomstring(percentholdout, nrow(laaccident))
  
  tmp.data <- cbind(laaccident,s)
  tmp.response <- (cbind(laaccident$Severity,s))
  holdout <- subset(tmp.data, s==1)[,1:length(laaccident)]
  holdout.response <- subset(tmp.response, s==1)[,1]
  train <- subset(tmp.data, s==0)[,1:length(laaccident)]
  
  sizeholdout <- dim(holdout)[1]
  sizetrain <- dim(train)[1]
  
  #ntree and mtry finalized after running the model individually
  finalrfmodel <- randomForest(Severity ~ ., train, ntree = mintreerf, mtry = optimalmtry, importance = T )
  rfpred <- predict(finalrfmodel, newdata = holdout)
  
  rfpred.train <- predict(finalrfmodel, newdata = train)
  
  predictionaccuracy[i,3] <- sum(diag(table(rfpred, holdout.response)))/sum(table(rfpred, holdout.response))
  trainingaccuracy[i,3] <- sum(diag(table(rfpred.train, train$Severity)))/sum(table(rfpred.train, train$Severity))
  
}

varImpPlot(finalrfmodel) #variable importance plot of Randomforest model

####################

#gbmboosting
library(gbm)
gbmboosting <- gbm(Severity ~ .,data =  train.data,distribution = "multinomial", n.trees=500, interaction.depth = 4)
gbmpred <- predict(gbmboosting, newdata = test.data, n.trees = 500, type = "response")
gbmpred <- as.factor(apply(gbmpred, 1, which.max))
summary(gbmboosting)
confusionMatrix(test.data$Severity,gbmpred)
plot(gbmboosting, i = 'Side')              #Partial dependence plot for side
plot(gbmboosting, i = 'Traffic_Signal')     #Partial dependence plot for traffic signal
plot(gbmboosting, i = 'Visibility.mi.')      #Partial dependence plot for visibility
plot(gbmboosting, i = 'Humidity...')          #Partial dependence plot for humidity 

#gradient boosting in for loop
library(gbm)
set.seed(17)
attach(laaccident)
for (i in 1:numholdout) {
  s <- randomstring(percentholdout, nrow(laaccident))
  
  tmp.data <- cbind(laaccident,s)
  tmp.response <- (cbind(laaccident$Severity,s))
  holdout <- subset(tmp.data, s==1)[,1:length(laaccident)]
  holdout.response <- subset(tmp.response, s==1)[,1]
  train <- subset(tmp.data, s==0)[,1:length(laaccident)]
  
  sizeholdout <- dim(holdout)[1]
  sizetrain <- dim(train)[1]
  
  boosting <- gbm(Severity ~ .,data =  train,distribution = "multinomial", n.trees=500, interaction.depth = 4)
  boostpred <- predict(boosting, newdata = holdout, n.trees = 500, type = "response")
  boostpred <- as.factor(apply(boostpred, 1, which.max))
  
  boostpredtrain <- predict(boosting, newdata = train, n.trees = 500, type = 'response')
  boostpredtrain <- as.factor(apply(boostpredtrain, 1, which.max))
  
  predictionaccuracy[i,4] <- sum(diag(table(boostpred, holdout.response)))/sum(table(boostpred, holdout.response))
  trainingaccuracy[i,4] <- sum(diag(table(boostpredtrain, train$Severity)))/sum(table(boostpredtrain, train$Severity))
  
}

##################

#ada boosting
library(ada)
set.seed(15)
adaboosting <- ada(Severity ~., train.data, iter = 50)
plot(adaboosting) #taking 45 as number of iteration
adapred <- predict(adaboosting, test.data)
confusionMatrix(test.data$Severity,adapred)

#ada boosting in for loop
library(ada)
set.seed(33)
attach(laaccident)
for (i in 1:numholdout) {
  s <- randomstring(percentholdout, nrow(laaccident))
  
  tmp.data <- cbind(laaccident,s)
  tmp.response <- (cbind(laaccident$Severity,s))
  holdout <- subset(tmp.data, s==1)[,1:length(laaccident)]
  holdout.response <- subset(tmp.response, s==1)[,1]
  train <- subset(tmp.data, s==0)[,1:length(laaccident)]
  
  sizeholdout <- dim(holdout)[1]
  sizetrain <- dim(train)[1]
  
  boostingmodelada <- ada(Severity ~ ., train, iter = 45)
  pred.boostada <- predict(boostingmodelada, holdout)
  
  pred.train.boostada <- predict(boostingmodelada, train)
  
  predictionaccuracy[i,5] <- sum(diag(table(pred.boostada, holdout.response)))/sum(table(pred.boostada, holdout.response))
  trainingaccuracy[i,5] <- sum(diag(table(pred.train.boostada, train$Severity)))/sum(table(pred.train.boostada, train$Severity))
  
}

###########################
#null model
library(caret)
set.seed(97)
attach(laaccident)
for (i in 1:numholdout) {
  s <- randomstring(percentholdout, nrow(laaccident))
  
  tmp.data <- cbind(laaccident,s)
  tmp.response <- (cbind(laaccident$Severity,s))
  holdout <- subset(tmp.data, s==1)[,1:length(laaccident)]
  holdout.response <- subset(tmp.response, s==1)[,1]
  train <- subset(tmp.data, s==0)[,1:length(laaccident)]
  
  sizeholdout <- dim(holdout)[1]
  sizetrain <- dim(train)[1]
  
  nullmodel <- nullModel(y = train$Severity, type = 'class') 
  
  pred.nullmodel <- predict(nullmodel, holdout)
  
  pred.train.nullmodel <- predict(nullmodel, train)
  
  predictionaccuracy[i,6] <- sum(diag(table(pred.nullmodel, holdout.response)))/sum(table(pred.nullmodel, holdout.response))
  trainingaccuracy[i,6] <- sum(diag(table(pred.train.nullmodel, train$Severity)))/sum(table(pred.train.nullmodel, train$Severity))
  
}

#finding the average prediction and training accuracy

meanpredictionaccuracy <- c()

for (k in 1:nmodel) {
  meanpredictionaccuracy[k] <- mean(predictionaccuracy[, k])
  
}

meanpredictionaccuracy  #gives the mean prediction accuracy of all the models
max(meanpredictionaccuracy) #gives the maximum prediction accuracy out of all models
which.max(meanpredictionaccuracy) #gives which model has the maximum prediction accuracy
#model 3 has the highest prediction accuracy (Randomforest)

meantrainingaccuracy <- c()

for (k in 1:nmodel) {
  meantrainingaccuracy[k] <- mean(trainingaccuracy[, k])
  
}

meantrainingaccuracy   #gives the mean training accuracy of all the models
max(meantrainingaccuracy) #gives the maximum training accuracy out of all models
which.max(meantrainingaccuracy) #gives which model has the maximum training accuracy
#model 3 has the highest training accuracy (Randomforest)
###################

dev.off()