\name{print.knncat}
\alias{print.knncat}
\title{Print a knncat classifier}
\description{Print the misclassification rate for a knncat classifier}
\usage{
\method{print}{knncat}(x, \dots)
}
\arguments{
\item{x}{Knncat object, from \link{knncat}}
\item{\dots}{Other arguments, currently ignored}
}               
\keyword{models}
\details{This prints the misclassification rate from the knncat classifier,
together with an indication as to whether it was based on a training or
test set.}
\value{None.}
\author{Samuel E. Buttrey, \email{buttrey@nps.edu}}
