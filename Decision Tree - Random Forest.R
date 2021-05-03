
#Student Name: Antonio Caruso
#Student Number: 19203608
#Project Title: CA Part 3
#Submission due: 24/07/2020
#Lecturer: John Kelly 
#############################

####

#1st Step: Read the file

Bank <- read.csv(file = "bank-full.csv", header = TRUE, sep = ";")

#2nd step: Factual Analysis on Variables
str(Bank)

#3rd Step checked whether there are NA or blanks
Bank[!complete.cases(Bank),] #no NAs,
colSums(Bank == "") # no blanks were detected


#4th Step Converted dependent variable Y into Factor

Bank$y <- as.factor(Bank$y)

#summary of dependent variable
summary(Bank$y) #39.922 actually responded no, 5,289 responded yes


#5th Step data split in training 70%, and test 30% 

dt <- sort(sample(nrow(Bank), nrow(Bank)*.7))

train <- Bank[dt,] 
#
test <- Bank[-dt,]  

nrow(train) # 31,647 observations
nrow(test) #13,564 observations

edit(train) #used to view the dataset


#6h step: built decision tree model on the dependent variable Y.  
set.seed(1234)
library(rpart)
Banktree <- rpart(y~., data = train, method = "class",control = rpart.control(minsplit = 20, minbucket = 7, maxdepth = 10, usesurrogate = 2, xval =10 ))
#20 min split for each value chosen as default value

print(Banktree) #the output displays duration and poutcome as variables used for the model
summary(Banktree) #to view details on each node, variable importance -> duration 61, poutcome 38

## Classification Tree Visualisation
library(rattle) 
library(tibble)
library(bitops)
library(rpart.plot)
library(RColorBrewer)

prp(Banktree, faclen = 0, cex = 0.9, extra = 2)
#View of fancy plot
fancyRpartPlot(Banktree)


#Another view with total count at each node
tot_count <- function(x, labs, digits, varlen)
{paste(labs, "\n\nn =", x$frame$n)}

prp(Banktree, faclen = 0, cex = 0.9, node.fun=tot_count)


#7th Step: printed the complexity parameter
printcp(Banktree) 

bestcp <- Banktree$cptable[which.min(Banktree$cptable[,"xerror"]),"CP"] #the best is 0.1


#8th step: used the pruning because the three might overfit the dataset, 
pruned <- prune(Banktree, cp= bestcp)

#and plotted it
prp(pruned, faclen= 0, cex = 0.8, extra = 1)


#9th step: reverse confusion matrix built(training data) for machine learning classification

conf.matrix <- table(train$y, predict(pruned, type = "class"))
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Pred", colnames(conf.matrix), sep = ":")
print(conf.matrix)  
#Output: 
#True Negative: 27,207
#False Positive: 687
#False Negative: 2439
#True positive 1314

#10 step checked the accuracy of the prediction

library(gmodels) 

CrossTable(conf.matrix)  #9% True Negative, 6% True positive
#

####11th Step Analysis of AUC 
library(ROCR)
test1 = predict(pruned, test, type = "prob")



#Stored Model Performance Scores
pred_test <-prediction(test1[,2],test$y)
# Calculated Area under Curve
perf_test <- performance(pred_test,"auc")
# Calculated True Positive and False Positive Rate
perf_test <- performance(pred_test, "tpr", "fpr")
# Plotted the ROC curve
plot(perf_test, col = "green", lwd = 1.5) 
#from the graph, we can see a AUC value of 0.50, FOR 100% correctness it should be 1

str(perf_test)

#Calculated KS statistics 
ks1.tree <- max(attr(perf_test, "y.values")[[1]] - (attr(perf_test, "x.values")[[1]]))
ks1.tree 
#0.49, not a very high value

##########################################

#### Random Forest applied to improve the efficiency of the model

library(randomForest)
library(gclus)
library(cluster)

#implemented the model with random forest algorithm

fit.r <- randomForest(y~., data = train,importance=TRUE)

#examined the result
fit.r   #4 variables tried at each split , OOB estimate error rate 9.42%


#finally, a look at the variable importance
varImpPlot(fit.r)


#Final AUC and Lift analysis  OF random forest
library(ROCR)
testR = predict(fit.r, test, type = "prob")
#Stored Model Performance Scores
pred_test_R <-prediction(testR[,2],test$y)

# Calculated Area under Curve
perf_test_R <- performance(pred_test_R,"auc")
perf_test_R

# CalculatedTrue Positive and False Positive Rate
perf_test_R <- performance(pred_test_R, "tpr", "fpr")

# Plot the ROC curve
plot(perf_test_R, col = "green", lwd = 1.5) 
#area of 0.90, the random forest model improved the results of the classification tree


#####End of script






