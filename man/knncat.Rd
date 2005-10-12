\name{knncat}
\alias{knncat}
\title{Build a knncat classifier}
\description{Build a knncat classifier, which is used for nearest-neighbor 
classification with categorical variables; continuous are permitted too.}
\usage{
knncat (train, test, k = c(1, 3, 5, 7, 9), xvals = 10, xval.ceil = -1, 
    knots = 10, prior.ind = 4, prior, permute = 10, permute.tail = 1, 
    improvement = .01, ridge = .003, once.out.always.out = FALSE, 
    classcol = 1, verbose = 0)
}
\arguments{
\item{train}{data frame of training data, with the correct classification 
in the classcol column}
\item{test}{data frame of test data (can be omitted). This should have
the correct classification in the classcol column, too.}
\item{k}{vector of choices for number of nn's. Default c(1, 3, 5, 7, 9).}
\item{xvals}{number of cross-validations to use to find the best model
size and number of nn's. Default 10.}
\item{xval.ceil}{Maximum number of variables to add. -1 = Use the smallest
number from any xval; 0 = use the smallest number from the
first xval; >= 0, use that.}
\item{knots}{vector of number of knots for numeric variables. Reused if
necessary. Default: 10 for each.}
\item{prior.ind}{Integer telling how to compute priors. 1 = estimated from
training set; 2 = all equal; 3 = supplied in "prior"; 4 = ignored. Default: 4.}
\item{prior}{Numeric vector, one entry per unique element in the training
set's classcol column, giving prior probabilities. Ignored unless
prior.ind = 3; then they're normalized to sum to 1 and each
entry must be strictly > 0.}
\item{permute}{Number of permutations for variable selection. Default: 10.}
\item{permute.tail}{A variable fails the permutation test if permute.tail or 
more permutations do better than the original. Default: 1.}
\item{improvement}{Minimum improvement for variable selection. Ignored unless
present and permute missing, or permute = 0; then default = .01.}
\item{ridge}{Amount by which to "ridge" the W matrix for numerical 
stability. Default: .003.}
\item{once.out.always.out}{if TRUE, a variable that fails a permutation test
or doesn't improve by enough is excluded from further consideration during 
that cross-validation run. Default FALSE.}
\item{classcol}{Column with classification in it. Default: 1.}
\item{verbose}{Controls level of diagnostic output. Higher numbers produce
more output, sometimes 'way too much. 0 produces no output; 1 gives progress 
report for xvals. Default: 1.}
}               
\keyword{models}
\details{A knncat classifier converts categorical labels into real numbers
(phi) so as to produce a good k-nearest neighbor classifier. Continuous 
variables are handled by means of knots, in a manner similar to the 
linear spline representation.
Variable selection is done by a permutation test, or by setting an
"improvement" cutoff; error rate estimation is done by cross-validation.
After the cross-validations are done, we choose the best value of k from
among those proposed and the "best" number of variables, then make one
more pass through all the data to estimate the phis.}
\value{A list of S3 class knncat, containing the following entries:
\item{cdata}{A vector with one entry for each of the columns of train,
except the classification column, with value 1 if that column was used
in the final classifier, and 0 otherwise.}
\item{phi}{A list with the phi's. Each element of the list has, as its name,
the name of a column of train; the values of the element are the phi's, and
the names of that element are the levels of the variable. For numeric
variables, these names are "knot.1", "knot.2" etc.}
\item{k}{The vector of k's to be tried, as passed in.}
\item{best.k}{The best k selected.}
\item{misclass.mat}{A matrix, number of classes * number of classes, whose
columns give the correct classifications and rows, the estimates.}
\item{prior.ind}{Method used to compute the prior, as passed in.}
\item{prior}{A numeric vector, one per class, giving the prior probabilties, 
as computed by the program according to prior.ind.}
\item{status}{Return value from the program. 0 = no error.}
\item{misclass.type}{Type of misclass.mat. "train" means misclass.rate came
from the training set; "test," from the test set.}
\item{train}{Name of training set at build time.}
\item{vars}{Vector of names of columns actually used in model.}
\item{knots.vec}{Vector of numbers of knots, as passed in.}
\item{build}{Named vector holding five of the arguments used at build time:
permute, improvement, ridge, once.out.always.out, and xvals}
\item{missing}{Vector of values with which to replace missing values. These
are the most common values for categorical variables, and the means for
continuous ones.}
\item{knot.values}{List of knot locations, one element for each continuous
variable.}
}
\references{Buttrey, S.E., Nearest-neighbor classification with categorical
variables, Comp. Stat. Data Analysis 28 (1998), 157-169.}
\examples{
library(MASS) # Load data set
syncat <- knncat (synth.tr, classcol=3)
\dontrun{syncat
Train set misclass rate: 12.8%
synpred <- predict (syncat, synth.tr, synth.te, train.classcol=3,
                    newdata.classcol=3)
table (synpred, synth.te$yc)
       
synpred 0   1  
      0 460  91
      1  40 409
#
# Or do the whole thing in one pass:
#
}
knncat (synth.tr, synth.te, classcol=3)
\dontrun{
Test set misclass rate: 13.1%
}
}
\author{Samuel E. Buttrey, \email{buttrey@nps.edu}}
