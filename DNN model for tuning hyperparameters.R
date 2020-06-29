#DNN Model Tuning of USPS data
#Author: Gyan Shashwat


#**************************** Begin ****************************
library(keras) #Importing keras pakage
#install_keras()
#library(tensorflow) #Importing tensorflow
library(tfruns) #Importing tfruns 
tensorflow::tf$random$set_seed(100) #setting seed for non repeated output

library(parallel)
library(MASS)
load("data_usps_digits.RData")
#y_train <- to_categorical(y_train)
#y_test <- to_categorical(y_test)

plot_digit <- function(index, data) { #funtion to plot the digits
  tmp <- (-data + 1) / 2
  # to convert back to original
  z <- matrix( data = as.numeric(data[index, 256:1]), 16, 16 )
  image(z[16:1,1:16], col = gray((1:100)/100),
        xaxt = "n", yaxt = "n")
}
# plot few example digits
library(foreach)
library(doParallel)
detectCores()
par(mfrow = c(1,3), mar = rep(1.5, 4)) #paramete set to flot 3 images on one layout
plot_digit(264, x_train) #plot image for 3
plot_digit(3924, x_train) #plot image for 8
plot_digit(7185, x_train) #plot image for 5

range(x_train) #printing range of x_train dataset
range(x_test) #printing range of x_test dataset

# normalize to 0 - 1
range_norm <- function(x, a = 0, b = 1) { #funtion to normalize input variables in the range of 0-1
  ( (x - min(x)) / (max(x) - min(x)) )*(b - a) + a }
x_train <- apply(x_train, 2, range_norm) #applying normalization function range_norm on training predictor variables 
x_test <- apply(x_test, 2, range_norm) #applying normalization function range_norm on training predictor variables 
range(x_train) #printing normalized range of x_train dataset
range(x_test) #printing range of normalized x_test dataset
# one-hot encoding of target variable
y_train<- to_categorical(y_train,num_classes = 10) # one-hot encoding of training target variable num_classes is 10 for (0-9)
y_test<- to_categorical(y_test,num_classes = 10)# one-hot encoding of testing target variable num_classes is 10 for (0-9)

# split the test data in two halves: one for validation
# and the other for actual testing
set.seed(19200276)
# there are 2007 images in x_test
val <- sample(1:nrow(x_test),1007) #smaple rows for validation data
test <- setdiff(1:nrow(x_test), val) # smaple rows for test data
x_val <- x_test[val,] # predictor variable for validation data
y_val <- y_test[val,] # response variable for validation data
x_test <- x_test[test,] #predictor variable for test data
y_test <- y_test[test,] #response variable for test data
# need these later
N <- nrow(x_train) 
V <- ncol(x_train)

# flags grid of nodes for 3 hidden layer and dropout
size1_set=c(256,128,64)
size2_set=c(256,128,64)
size3_set=c(256,128,64)
dropout_set=c(0,0.3,0.5,0.6)
lambda_set <- c(0, exp( seq(-6, -4, length = 9) ))
#running the model                                
runs <- tuning_run("tuning.R",
                   runs_dir = "runs_example_new_1",
                   flags = list(
                     dropout = dropout_set,
                     unit1 = size1_set,
                     unit2 = size2_set,
                     unit3 = size3_set,
                     lambda=lambda_set
                   ),sample = 0.3)
                     
                  

library(jsonlite) #importing jsonlite pakage
library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
stopCluster(cl)
registerDoSEQ()
read_metrics <- function(path, files = NULL)
  # 'path' is where the runs are --> e.g. "path/to/runs"
{
  path <- paste0(path, "/")
  if ( is.null(files) ) files <- list.files(path)
  n <- length(files)
  out <- vector("list", n)
  for ( i in 1:n ) {
    dir <- paste0(path, files[i], "/tfruns.d/")
    out[[i]] <- jsonlite::fromJSON(paste0(dir, "metrics.json"))
    out[[i]]$flags <- jsonlite::fromJSON(paste0(dir, "flags.json"))
    out[[i]]$evaluation <- jsonlite::fromJSON(paste0(dir, "evaluation.json"))
  }
  return(out)
}
plot_learning_curve <- function(x, ylab = NULL, cols = NULL, top = 3, span = 0.4, ...)
{
  # to add a smooth line to points
  smooth_line <- function(y) {
    x <- 1:length(y)
    out <- predict( loess(y ~ x, span = span) )
    return(out)
  }
  matplot(x, ylab = ylab, xlab = "Epochs", type = "n", ...)
  grid()
  matplot(x, pch = 19, col = adjustcolor(cols, 0.3), add = TRUE)
  tmp <- apply(x, 2, smooth_line)
  tmp <- sapply( tmp, "length<-", max(lengths(tmp)) )
  set <- order(apply(tmp, 2, max, na.rm = TRUE), decreasing = TRUE)[1:top]
  cl <- rep(cols, ncol(tmp))
  cl[set] <- "deepskyblue2"
  matlines(tmp, lty = 1, col = cl, lwd = 2)
}


out <- read_metrics("runs_example_new")
# extract validation accuracy and plot learning curve
acc <- sapply(out, "[[", "val_accuracy")
plot_learning_curve(acc, col = adjustcolor("black", 0.3), ylim = c(0.85, 1),
                    ylab = "Val accuracy", top = 3)

# all flag value result object
res <- ls_runs(metric_val_accuracy > 0.92,
               runs_dir = "runs_example_new", order = metric_val_accuracy)

res <- res[,c(2,4,8:14)]
res[1:10,]

#******* 
