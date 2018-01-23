library(keras)
library(tensorflow)
library(ggplot2)

use_session_with_seed(1)

### Reads dataset 
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", 
                 sep = ",", header = FALSE, stringsAsFactors=FALSE) %>% as.matrix

### Removes ID column
data <- data[,-1]

### Generates random indices for training set
index <- sample(1:569, 480)

### Creates and transforms training set
train.data <- data[index,]
train.x <- train.data[, 2:31 ] %>%
  as.numeric %>%
  array(dim = c(480, 30))

train.y <- train.data[, 1]
train.y[train.y[] == "B"] <- 0
train.y[train.y[] == "M"] <- 1

train.y <- to_categorical(train.y) %>%
  as.array()
train.y <- train.y[, 2]

### Creates and transforms test set
test.data <- data[-index,]
test.x <- test.data[,2:31] %>%
  as.numeric %>%
  array(dim = c(89, 30))

test.y <- test.data[, 1]
test.y[test.y[] == "B"] <- 0
test.y[test.y[] == "M"] <- 1

test.y <- to_categorical(test.y) %>%
  as.array()
test.y <- test.y[, 2]

###Normalizes training/test sets

for(i in 1:30){
  train.x[,i] <- scale(train.x[,i])
}

for(i in 1:30){
  test.x[,i] <- scale(test.x[,i])
}

rm(data, test.data, train.data, i, index) #Removes used files

### Creates sequential model - MLP

ann.32 <- keras_model_sequential() 

weights <- initializer_random_normal(mean = 0, stddev = 0.1, seed = NULL)

ann.32 %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = 30, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

adam <- optimizer_adam(lr = 0.0003)

ann.32 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.32 <- ann.32 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 1
)

plot(history.32)

prediction.32 <- predict_classes(ann.32, test.x) %>% as.matrix

#Creates prediction file
result <- data.frame(prediction.32, test.y) %>%
  write.table("result_32.txt", sep=",") 

# Prints test and train accuracy
test.result.32 <- evaluate(ann.32, test.x, test.y, verbose = 0)
train.result.32 <- evaluate(ann.32, train.x, train.y, verbose = 0)


########### Experimenting with different activation functions ################

## Sigmoid
ann.sigmoid <- keras_model_sequential() 

ann.sigmoid %>% 
  layer_dense(units = 32, activation = 'sigmoid', input_shape = 30, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.sigmoid %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.sigmoid <- ann.sigmoid %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 1
)

prediction.sigmoid <- predict_classes(ann.sigmoid, test.x) %>% as.matrix

#Creates prediction file
result.sigmoid <- data.frame(prediction.sigmoid, test.y) %>%
  write.table("result_sigmoid.txt", sep=",") 

# Prints test and train accuracy
evaluate(ann.sigmoid, test.x, test.y, verbose = 0)
evaluate(ann.sigmoid, train.x, train.y, verbose = 0)


# Creates table for ploting ReLU/Sigmoid accuracy/loss
relu.sig.loss <- matrix(nrow = 20, ncol = 3)

relu.sig.loss[,1] <- rep(1:20)
relu.sig.loss[,2] <- history.32$metrics$loss
relu.sig.loss[,3] <- history.sigmoid$metrics$loss
colnames(relu.sig.loss) <- c("Epoch", "ReLU","Sigmoid")


plot(relu.sig.loss[,2], type = "o", col = "coral2", ylim = c(0, 1),  
     ylab = "Loss", xlab = "Epoch")

lines(relu.sig.loss[,3], type = "o", col = "blue")

legend(14, 1, c("ReLU", "Sigmoid"), 
       lty = c(1, 1), col = c("Coral2", "Blue"))


relu.sig.acc <- matrix(nrow = 20, ncol = 3)

relu.sig.acc[,1] <- rep(1:20)
relu.sig.acc[,2] <- history.32$metrics$acc
relu.sig.acc[,3] <- history.sigmoid$metrics$acc
colnames(relu.sig.acc) <- c("Epoch", "ReLU","Sigmoid")


plot(relu.sig.acc[,2], type = "o", col = "coral2", ylim = c(0.2, 1),  
     ylab = "Accuracy", xlab = "Epoch")

lines(relu.sig.acc[,3], type = "o", col = "blue")

legend(14, 0.5, c("ReLU", "Sigmoid"), 
       lty = c(1, 1), col = c("Coral2", "Blue"))

########### Experimenting with different number of nodes ################

#### 4 node hidden layer model
ann.4 <- keras_model_sequential() 

ann.4 %>% 
  layer_dense(units = 4, activation = 'relu', input_shape = 30, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.4 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.4 <- ann.4 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.4 <- predict_classes(ann.4, test.x) %>% as.matrix

#Creates prediction file
result.4 <- data.frame(prediction.4, test.y) %>%
  write.table("result_4.txt", sep=",") 

# Prints test and train accuracy
test.result.4 <- evaluate(ann.4, test.x, test.y, verbose = 0)
train.result.4 <- evaluate(ann.4, train.x, train.y, verbose = 0)


#### 8 node hidden layer model
ann.8 <- keras_model_sequential() 

ann.8 %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = 30, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.8 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.8 <- ann.8 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.8 <- predict_classes(ann.8, test.x) %>% as.matrix

#Creates prediction file
result.8 <- data.frame(prediction.8, test.y) %>%
  write.table("result_8.txt", sep=",") 

# Prints test and train accuracy
test.result.8 <- evaluate(ann.8, test.x, test.y, verbose = 0)
train.result.8 <- evaluate(ann.8, train.x, train.y, verbose = 0)


#### 16 node hidden layer model
ann.16 <- keras_model_sequential() 

ann.16 %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = 30, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.16 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.16 <- ann.16 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.16 <- predict_classes(ann.16, test.x) %>% as.matrix

#Creates prediction file
result.16 <- data.frame(prediction.16, test.y) %>%
  write.table("result_16.txt", sep=",") 

# Prints test and train accuracy
test.result.16 <- evaluate(ann.16, test.x, test.y, verbose = 0)
train.result.16 <- evaluate(ann.16, train.x, train.y, verbose = 0)


#### 64 node hidden layer model
ann.64 <- keras_model_sequential() 

ann.64 %>% 
  layer_dense(units = 64, activation = 'relu', input_shape = 30, 
              bias_initializer = 'ones',
              kernel_initializer = weights,
              trainable = TRUE,
              name = 'layer1') %>% 
  layer_dense(units = 1, activation = 'sigmoid',
              name = 'outlayer')

ann.64 %>% compile(
  optimizer = adam,
  loss = 'binary_crossentropy', 
  metrics = 'accuracy'
)

history.64 <- ann.64 %>% fit(
  train.x,
  train.y,
  epochs = 20,
  validation_split = 0.1, 
  verbose = 0
)

prediction.64 <- predict_classes(ann.64, test.x) %>% as.matrix

#Creates prediction file
result.64 <- data.frame(prediction.64, test.y) %>%
  write.table("result_64.txt", sep=",") 

# Prints test and train accuracy
test.result.64 <- evaluate(ann.64, test.x, test.y, verbose = 0)
train.result.64 <- evaluate(ann.64, train.x, train.y, verbose = 0)

table(prediction.64, test.y)

# Creates bar plot for testing data
nodes.test <- matrix(ncol = 2, nrow = 5)
colnames(nodes.test) <- c("Nodes","Accuracy")
nodes.test[,1] <- c("4","8","16","32","64")

nodes.test[,2] <- c(round(test.result.4$acc, 2)*100,
                    round(test.result.8$acc, 2)*100,
                    round(test.result.16$acc, 2)*100,
                    round(test.result.32$acc, 2)*100,
                    round(test.result.64$acc, 2)*100)

nodes.test <- data.frame(nodes.test)
nodes.test$Nodes <- factor(nodes.test$Nodes, c(4, 8, 16, 32, 64))

ggplot(nodes.test, aes(x = Nodes, y = Accuracy)) +
  geom_bar(stat = "identity") +
  ggtitle("Testing Accuracy")


# Creates bar plot for training data
nodes.train <- matrix(ncol = 2, nrow = 5)
colnames(nodes.train) <- c("Nodes","Accuracy")
nodes.train[,1] <- c("4","8","16","32","64")

nodes.train[,2] <- c(round(train.result.4$acc, 2)*100,
                     round(train.result.8$acc, 2)*100,
                     round(train.result.16$acc, 2)*100,
                     round(train.result.32$acc, 2)*100,
                     round(train.result.64$acc, 2)*100)

nodes.train <- data.frame(nodes.train)
nodes.train$Nodes <- factor(nodes.train$Nodes, c(4, 8, 16, 32, 64))

ggplot(nodes.train, aes(x = Nodes, y = Accuracy)) +
  geom_bar(stat = "identity")+
  ggtitle("Training Accuracy")


### prints confusion matrices 
table(prediction.4, test.y)
table(prediction.8, test.y)
table(prediction.16, test.y)
table(prediction.32, test.y)
table(prediction.64, test.y)

table(prediction.sigmoid, test.y)












