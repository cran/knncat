\name{predict.knncat}
\alias{predict.knncat}
\title{Prediict on a knncat classifier}
\description{Produce predictions for a knncat classifier}
\usage{
\method{predict}{knncat}(object, train, newdata, 
train.classcol=1, newdata.classcol=1, return.classes=TRUE,
more=FALSE, verbose = 0, \dots)
}
\arguments{
\item{object}{Knncat object, from \link{knncat}}
\item{train}{Training set used to build classifier}
\item{newdata}{New data on which to make predictions}
\item{train.classcol}{Column number for classification in training set.
Default: 1}
\item{newdata.classcol}{Column number for classification in newdata set.
Default: 1. If <= 0, new data has no classifications.}
\item{return.classes}{Logical; if TRUE, return a vector of classifications
of the newdata set. Default: TRUE}
\item{more}{Logical; if TRUE, also print error rate. Default: FALSE}
\item{verbose}{Level of verbosity for debugging. Default: 0}
\item{\dots}{Other arguments, currently ignored}
}               
\keyword{models}
\details{This prints the misclassification rate from the knncat classifier,
together with an indication as to whether it was based on a training or
test set.}
\value{None.}
\author{Samuel E. Buttrey, \email{buttrey@nps.edu}}
