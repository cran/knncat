/*
**      donnwrap: wrapper to call do_nn from R
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "utils.h"
#include "ord.h"
#include "donn.h"

#ifdef USE_R_ALLOC
extern void Rprintf ();
void do_nothing();
#define free do_nothing
#endif

#ifdef CALL_FROM_R
#define Slong int
#else
#define Slong long
#endif

void donnwrap (
          double *traindata,  Slong *trainrows,  Slong *traincols,
          double *testdata,  Slong *testrows,  Slong *testcols,
          Slong *cats_in_var, Slong *cum_cats_this_subset,
          double *cdata, double *phidata, double *priordata,
          Slong *number_of_classes, Slong *increase,
          double *in_knots, double *error_rate,
          Slong *best_k, Slong *return_classes, Slong *classes,
          Slong *verbose, Slong *status)
{
long i;
MATRIX *training, *test, *c, *cost, *misclass_mat, *prior;
long quit, how_many_ks, theyre_the_same, xval_lower, xval_upper;
long knot_ctr, number_of_vars;

double **knots;

training       = make_matrix (*trainrows, *traincols, 
                        "Training set", REGULAR, FALSE);
training->data = traindata;
test           = make_matrix (*testrows, *testcols, 
                        "Test set", REGULAR, FALSE);
test->data     = testdata;
c              = make_matrix (1, *traincols, "C", REGULAR, TRUE);
for (i = 0; i < *traincols; i++)
    c->data[i] = 1.0;
cost           = (MATRIX *) NULL;
misclass_mat   = (MATRIX *) NULL;
prior          = make_matrix (*number_of_classes, *number_of_classes,
                     "Prior", REGULAR, TRUE);
for (i = 0; i < *number_of_classes; i++)
    *SUB (prior, i, i) = priordata[i];

quit           = FALSE;
how_many_ks    = 1L;
xval_lower     = 2L;
xval_upper     = 1L;
theyre_the_same = (long) FALSE;

number_of_vars = *traincols - 1;
alloc_some_double_pointers (&knots, number_of_vars);

knot_ctr = 0;
for (i = 0; i < number_of_vars; i++)
{
    if (increase[i] == NUMERIC)
    {
        knots[i] = &(in_knots[knot_ctr]);
        knot_ctr += cats_in_var[i];
    }
} /* end "for i" loop. */

do_nn (&quit, training, test, c, best_k, &how_many_ks,
           &theyre_the_same, phidata, cats_in_var, 
           cum_cats_this_subset,
           knots,
           (MATRIX *) NULL, prior, error_rate,
           (MATRIX *) NULL, return_classes, classes,
           &xval_lower, &xval_upper, (long *) NULL,
           increase, number_of_classes, verbose);

quit = TRUE;
do_nn (&quit, training, test, c, best_k, &how_many_ks,
           &theyre_the_same, phidata, cats_in_var, 
           cum_cats_this_subset,
           knots,
           (MATRIX *) NULL, prior, error_rate,
           (MATRIX *) NULL, return_classes, classes,
           &xval_lower, &xval_upper, (long *) NULL,
           increase, number_of_classes, verbose);

free (c->data);
free (prior->data);
free (knots);

} /* end "donnwrap" */
