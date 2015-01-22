/*
** ord.c: Classification of ordinal data
*/
/*
** Enormous and vital globals: "am_i_in" (increase now local)
** Also xval_indices, xval_lower, xval_upper. Not to mention
** original_margin_holder, cum_cats_this_subset...
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#ifdef USE_R_ALLOC
void do_nothing();
#define free do_nothing
#endif

#ifdef CALL_FROM_R
#define Slong int
#else
#define Slong long
#endif

/* These two are for "getrusage," for adding up system usage */
/**
#include <sys/time.h>
#include <sys/resource.h>
**/

#include "matrix.h"
#include "utils.h"
#include "ord.h"
#include "ordfuncs.h"
#include "donn.h"
#include "dodisc.h"
#include "R.h"

Slong Slong_TRUE  = (Slong) TRUE;
Slong Slong_FALSE = (Slong) FALSE;
long zero       = 0L;
long one        = 1L;

/* Definition of Lapack routines to be called via R's dll */
void F77_NAME(dsyev)
(char *the_letter_V, char *the_letter_U, long *nrow, double *data,
        long *nrow2, double *edata, double *wdata, long *wlen, 
        long *LAPACK_status);

void F77_NAME(dsygv)
(long *problem_type, char *the_letter_V, char *the_letter_U, long *nrow,
            double *data, long *nrow2, double *wdata, 
            long *wrow, double *edata, double *work, long *workl,
            long *LAPACK_status);

/* Structure for usage stats via getrusage */
/*
struct rusage usage_struct;
*/

long xval_ctr, xval_lower, xval_upper;
long xval_ceiling_ind;
long *xval_indices = (long *) NULL;

MATRIX *global_copy_of_w;
MATRIX *global_copy_of_u;

/*
** Training data matrix ptr and number of rows.
*/
MATRIX *training;
unsigned long train_n = 0L;
unsigned long train_n_effective = 0L;
long train_n_long; /* for call to dsort */
/*
** Test data matrix ptr and number of rows.
*/
MATRIX *test;
unsigned long test_n  = 0L;

/* extern int errno; */

void constraint (), objective ();
MATRIX *phi = (MATRIX *) NULL;

long *am_i_in;               /* Keeps track of what variables are in.  */
long am_i_in_ctr;            /* Keeps track of variable rankings.      */
Slong *cum_cats_this_subset;  /*                                        */
Slong *cats_in_var;           /* Ith entry: num of levels in var i      */
long current_cat_total;
long currently_ordered;      /* Number of variables in the current subset */
                             /* which are ordered                         */
long currently_numeric;
long orderable_cats;         /* Number of cats in the current subset's    */
                             /* ordered variables                         */
int all_unordered;
int all_numeric;
Slong theyre_the_same = FALSE; /* Are training and test sets identical? */
int delete_this_variable_forever = TRUE;

int classification = CLASSIFICATION;
long *once_out_always_out;

double **knots;

MATRIX *prior;
double *priordata;

MATRIX *cost = (MATRIX *) NULL;
int use_cost = FALSE;
char cost_file_name[80];
FILE *cost_file;

double eigen_tolerance = 1.0e-4;
/*
** "Original" refers to the unpermuted matrices; without "original"
** they're the permuted ones. "Eigenvec_ptr" and "eigenval_ptr" point
** to whichever of "original" or not is current. "Last" holds the info
** from the last time through, in case no variable is added this time
** through. "Best" holds the best eigenvalues and vectors from among all
** the candidates on this round. ("Best" means "corresponding to the
** largest of the second eigenvalues.")
**
** Oh, yeah; "imaginary" and "beta" are needed when ridge = 0 because
** of degeneracy...
*/
MATRIX *original_eigenvectors, *eigenvectors, 
       *original_eigenvalues, *eigenvalues_real, 
       *eigenvec_ptr, *eigenval_ptr,
       *best_eigenvector, *best_w_inv_m_mat,
       *last_eigenvec_mat, *last_eigenval_mat;
MATRIX *u, *w;                    /* Need to be global for ordered case. */
long   best_variable;
double best_eigenvalue;
double relevant_eigenvalue, largest_eigenvalue;
long largest_eigenvalue_num;
long relevant_eigenvalue_num, original_relevant_num, 
     last_relevant_num, best_relevant_num;
double improvement;

MATRIX *original_margin_holder, *permuted_margin_holder;
/*
** Number_of_vars is the number of predictors, and therefore doesn't
** include the class identifier; both test and train have (number_of_vars
** + 1) columns.
*/
long number_of_vars;
Slong *increase;

Slong number_of_classes = 0L;    /* Number of classes. You guessed that. */

long permute, permute_tail;
/*
** "K" is the number of nearest neighbors; "slots" (the number of neighbors we
** look at ( > k to deal with ties better) is set in do_nn. K is a vector so 
** that we can look a set of numbers of NNs at a time.
*/
long k_len, slots;

double ridge;

double missing_max = MISSING_MAX;
double *missing_values;
long xvals;
double smallest_misclass_error;
long smallest_misclass_dim, smallest_misclass_k;
MATRIX *misclass_mat;

Slong verbose = 0;

/*============================= ord ==================================*/
void ord (double *traindata, Slong *trainrows, Slong *traincols,
          double *testdata,  Slong *testrows,  Slong *testcols,
          Slong *test_classes,
          double *cdata, double *phidata,
          Slong *in_number_of_classes, Slong *in_xvals,
          Slong *in_increase, Slong *in_permute, Slong *in_permute_tail,
          double *in_ridge, double *in_knots,
          Slong *in_k_len, Slong *in_k, Slong *in_best_k,
          Slong *in_classification, double *in_improvement, 
          Slong *in_cats_in_var, Slong *in_number_in_class, 
          double *misclass_data, Slong *xval_ceiling_ind, 
          Slong *in_once_out_always_out,
          Slong *in_prior_ind, double *in_priordata,
          Slong *in_verbose, Slong *status)
{
long i, j;                   /* Iteration counters.                    */
long cat_ctr;                /*                                        */
long var_ctr;                /*                                        */
long permute_ctr;            /*                                        */
long permute_len;            /*                                        */
long *cum_cats_before_var;   /* Ith : # of categories in vars 1-(i-1)  */
Slong *cum_cats_ptr;         /*                                        */
long *number_in_class;       /*                                        */
long *number_in_class_ptr;   /*                                        */
long total_number_of_cats;   /* Number of levels in all categories     */
long *permute_indices;       /*                                        */
MATRIX *c;                   /*                                        */
double *misclass_rate;       /*                                        */
Slong *k, best_k;
long row_ctr;
Slong prior_ind;
long dimension = 0L;
int first_time_through;
int do_the_omission = 0;
int do_nn_result, get_seq_result;
long xval_ceiling;
Slong *return_classes = (Slong *) NULL;

#define null_mat     (MATRIX *) NULL
/* Matrices for discrimination stuff. */
MATRIX *original_w_inv_m_mat = null_mat;
MATRIX *last_w_inv_m_mat = null_mat;

/* Knot stuff. "Cats in var[i]" already tells us the number of knots in
** the ith-variable if it's continuous. "Knots[i]" is a vector of that
** length that holds the knots, and "knots[i][j]" is the value of the
** jth knot on the ith variable. 
*/

long *holes;

long knot_ctr;

/*
** "Xval_result" is an matrix that holds the results of cross-validation. 
** It is (number_of_vars) by length(k_vector): many rows will not be used.
** The jth row holds info about the classifier built on the first j rows.
**
*/
MATRIX *xval_result;

permute            = (long) *in_permute;
permute_tail       = (long) *in_permute_tail;
ridge              = *in_ridge;               /* this is double */
k_len              = (long) *in_k_len;
best_k             = (Slong) *in_best_k;

/* Loop over the vector of nn's to fill k */
alloc_some_Slongs (&k, k_len);
for (i = 0; i < k_len; i++)
    k[i] = (Slong) in_k[i];
xvals              = (long) *in_xvals;
classification     = (long) *in_classification;
improvement        = *in_improvement;         /* this is double */

number_of_vars    = (long) *traincols - 1;
/* Loop over the set of columns to fill these two */
alloc_some_Slongs (&cats_in_var, number_of_vars);
alloc_some_Slongs (&increase, number_of_vars);
for (i = 0; i < number_of_vars; i++)
{
    cats_in_var[i] = in_cats_in_var[i];
    increase[i]    = in_increase[i];
}
/* Loop over the set of classes to fill number_in_class */
number_of_classes = (Slong) *in_number_of_classes;
alloc_some_longs (&number_in_class, (long) number_of_classes);
for (i = 0; i < number_of_classes; i++)
    number_in_class[i] = in_number_in_class[i];

priordata          = in_priordata;            /* this is double */
prior_ind          = (Slong) *in_prior_ind;
verbose            = (Slong) *in_verbose;
once_out_always_out = (long *) in_once_out_always_out;
*status            = (Slong) 0;


all_unordered     = all_numeric = FALSE;
train_n           = (unsigned long) *trainrows;
train_n_long      = (long) *trainrows;
test_n            = (unsigned long) *testrows;

if (prior_ind == IGNORED)
    prior = (MATRIX *) NULL;
else
    prior = make_matrix (number_of_classes, number_of_classes, "Priors", 
                         REGULAR, ZERO_THE_MATRIX);

alloc_some_longs (&permute_indices, train_n);
alloc_some_longs (&xval_indices, train_n);
alloc_some_longs(&cum_cats_before_var, number_of_vars);
alloc_some_Slongs(&cum_cats_this_subset, number_of_vars);
alloc_some_doubles(&missing_values, number_of_vars);
for (cat_ctr = 0L; cat_ctr < number_of_vars; cat_ctr++)
{
    cum_cats_before_var [cat_ctr] = 0L;
    cum_cats_this_subset[cat_ctr] = (Slong) 0;
}
alloc_some_longs(&am_i_in, number_of_vars);
training = make_matrix ((long) train_n, number_of_vars + 1, 
                        "Training set", REGULAR, FALSE);
training->data = traindata;
if (test_n > 0) 
{
    test = make_matrix ((long) test_n, number_of_vars + 1, 
                        "Test set", REGULAR, FALSE);
    test->data = testdata;
}

alloc_some_doubles (&misclass_rate, k_len);
c = make_matrix (1L, number_of_vars, "C", REGULAR, FALSE);
c->data = cdata;

if (all_numeric == TRUE)
    do_the_omission = FALSE;


/*
** Count the number of categories. For the moment, require that the
** categories be integers; let's just find the biggest one and assume
** that the number of categories is that thing plus one.
*/
total_number_of_cats = 0;

for (j = 1; j <= number_of_vars; j++)
{
/*
** If this variable is numeric, decide how many knots it should have.
** This number is in "cats_in_var".
*/
    am_i_in[j-1] = CURRENTLY_OUT;
    if ((increase[j-1] == DECREASING  || increase[j-1] == INCREASING)
            && classification == DISCRIMINATION)
    {
        Rprintf ( "Warning: making variable %ld unordered\n", j-1);
        increase[j-1] = UNORDERED;
    }
    total_number_of_cats += cats_in_var[j-1];
    if (j == 1)
        {/* do nothing */}
    else
        if (j == 2)
            cum_cats_before_var[1] = cats_in_var[0];
        else
           cum_cats_before_var[j-1] = cum_cats_before_var[j-2]
                                      + cats_in_var[j-1];
} /* end "for j" looping over variables. */
/*
** The "margin holder" matrix is (number of classes + 1) by the total number
** of categories. The ith row has the marginal totals (class = i) for
** each level of each class. Here "marginal totals" means the number
** of rows in the associated class and category, for a categorical
** variable, or the sum of (x_i - the knot associated with this category),
** for a continuous one. The final row holds the totals of these things
** over all classes.
*/
/***
original_margin_holder = make_matrix ((long) number_of_classes + 1, 
                         total_number_of_cats, "Margin holder", 
                         REGULAR, ZERO_THE_MATRIX);
permuted_margin_holder = make_matrix ((long) number_of_classes + 1, 
                         total_number_of_cats, "Permuted holder", 
                         REGULAR, ZERO_THE_MATRIX);
***/

alloc_some_double_pointers (&knots, number_of_vars);

knot_ctr = 0;
for (i = 0; i < number_of_vars; i++)
{
    if (all_numeric || increase[i] == NUMERIC)
    {
        knots[i] = &(in_knots[knot_ctr]);
        knot_ctr += cats_in_var[i];
    }
} /* end "for i" loop. */

cum_cats_ptr = cum_cats_this_subset;
number_in_class_ptr = number_in_class;
last_eigenval_mat = make_matrix (0L, 0L, "Last Val", REGULAR, FALSE);
last_eigenvec_mat = make_matrix (0L, 0L, "Last Vec", REGULAR, FALSE);
if (classification == DISCRIMINATION)
    last_w_inv_m_mat = make_matrix (0L, 0L, "Last W-inv-M", REGULAR, FALSE);
    if (last_w_inv_m_mat == (MATRIX *) NULL) { /* do nothing */ }

xval_result = make_matrix (number_of_vars, k_len, "Xval results",
                           REGULAR, ZERO_THE_MATRIX);

xval_lower = 0L;
if (xvals == 1)
{
    xval_upper = 0L;
    permute_len = train_n;
    train_n_effective = train_n;
}
else
{
    xval_upper = train_n / xvals;
    permute_len = train_n - (train_n / xvals);
    train_n_effective = train_n - (train_n / xvals);
}

/*=================== ENORMOUS FOR LOOP =======================*/
genprm (xval_indices, (int) train_n);
for (xval_ctr = 0L; xval_ctr < xvals; xval_ctr++)
{
/*
** Fill "permute_indices" with the indices of all the training set
** items that are in this cross-validation group.
*/
if (verbose > 1)
    Rprintf ("Top of xval loop, this is %i of %i\n", xval_ctr, xvals);
permute_ctr = 0L;
for (row_ctr = 0; row_ctr < train_n; row_ctr++)
{
    if (row_ctr >= xval_lower && row_ctr < xval_upper)
        continue;
    permute_indices[permute_ctr++] = row_ctr;
}
/*
** Loop through each categorical, indexed by "i".
*/
i = (long) -1;

dimension = 1L;
best_variable = -1L;
first_time_through = TRUE;
am_i_in_ctr = ALWAYS_IN;

get_seq_result = get_sequence_of_solutions ((long) -1,
    number_of_vars, permute, permute_len, permute_indices, improvement, 
    k_len, k, cats_in_var, cum_cats_ptr, c,
    &original_eigenvectors, &original_w_inv_m_mat,
    number_in_class_ptr, xval_ctr,
    xval_result, &xval_ceiling,
    cost, prior, prior_ind, number_of_classes,
    misclass_rate, do_the_omission, increase, once_out_always_out);

if (get_seq_result == FALSE)
{
    Rprintf ("Panic! Get_seq returned FALSE\n");
/*
    print_matrix (global_copy_of_w, (0x10) | (0x40));
    print_matrix (global_copy_of_u, (0x10) | (0x40));
    print_matrix (original_eigenvectors, (0x10) | (0x40));
*/
    *status = (Slong) -1;
    return;
}

xval_lower += train_n / xvals;
xval_upper += train_n / xvals;

} /* end enormous "for" loop */

if (dimension == 0)
{
    if (verbose > 1)
        Rprintf ("Hmmmm....no variable made it in. Guess the biggest?\n");
    *status  = (Slong) -2;
    return;
}

/*
** Replace entries in the second and higher columns of the test and training
** matrices with the relevant eigenvalues. We get these by looking into the
** training matrix's spot and finding the relevant value, then looking into
** the eigenvector matrix at the correct column and extracting what you 
** find there.
** HERE WE ASSUME THAT THE CATEGORIES START AT 0 AND GO UP BY 1's!
*/

if (verbose > 1)
{
    Rprintf ("Xval result is...\n");
    print_matrix (xval_result, 8);
    Rprintf ("Calling matrix_min with ceiling %ld\n", xval_ceiling);
}

smallest_misclass_error =  matrix_min (xval_result, &smallest_misclass_dim, 
                                       &smallest_misclass_k, xval_ceiling);

if (verbose > 1)
    Rprintf ("Best rate is with %li dims and %li nn's\n",
                     smallest_misclass_dim, (long) k[smallest_misclass_k]);

xval_indices = (long *) NULL;
train_n_effective = train_n;

if (verbose > 0)
    Rprintf ("Now making final call with all data\n");

get_seq_result = get_sequence_of_solutions (smallest_misclass_dim + 1,
    number_of_vars, /* permute = */ 0L, permute_len, permute_indices, 
    improvement, 
    /* k_len = */ 1L, &(k[smallest_misclass_k]), cats_in_var, cum_cats_ptr, c,
    &original_eigenvectors, &original_w_inv_m_mat,
    number_in_class_ptr, /* xval_ctr = */ 1L,
    (MATRIX *) NULL, &xval_ceiling,
    cost, prior, prior_ind, number_of_classes,
    misclass_rate, do_the_omission, increase, once_out_always_out);
if (get_seq_result == FALSE)
{
    Rprintf ("Panic! Get_seq returned FALSE\n");
    *status = (Slong) -1;
    return;
}

if (verbose >= 3)
    print_matrix (c, 8);

if (test_n > 0)
    insert_missing_values (test, missing_max, missing_values);

misclass_mat = make_matrix (number_of_classes, number_of_classes, 
                     "Misclass mat", REGULAR, FALSE);
misclass_mat->data = misclass_data;

if (classification == CLASSIFICATION)
{
    phi = make_matrix (1L, original_eigenvectors->nrow, "My row", REGULAR, 
                                 ZERO_THE_MATRIX);
    matrix_extract (original_eigenvectors, ROW, 
                    original_eigenvectors->nrow - 2,
                    phi, FALSE);

    if (verbose >= 3) 
    {
        Rprintf ("Optimal phi is...\n");
        print_matrix (phi, 8);
    }
    for (i = 0; i < original_eigenvectors->nrow; i++)
        phidata[i] = phi->data[i];
    
    if (prior_ind != IGNORED)
        for (i = 0; i < number_of_classes; i++)
            priordata[i] = *SUB (prior, i, i);
    

    best_k = k[smallest_misclass_k];
    do_nn_result = TRUE;

/*
** If there's no test set, call do_nn once more with theyre_the_same
** = TRUE to get a training set error rate. If there is a test set,
** use that.
*/
    if (test_n == 0)
    {
        return_classes = &Slong_FALSE;
        do_nn_result = do_nn (&Slong_FALSE, training, training,
                     c, &best_k, &one, 
                     &Slong_TRUE, /* theyre_the_same */
                     phi->data, cats_in_var,
                     cum_cats_ptr, knots, cost, &prior_ind, prior, 
                     misclass_rate, misclass_mat, 
                     return_classes, test_classes, 
                     &one, &zero, (long *) NULL, /* <- xval stuff */
                     increase, &number_of_classes, &verbose);
    }
    else
    {
        return_classes = &Slong_TRUE;
        do_nn_result = do_nn (&Slong_FALSE, training, test,
                     c, &best_k, &one, 
                     &Slong_FALSE, /* theyre_the_same */
                     phi->data, cats_in_var,
                     cum_cats_ptr, knots, cost, &prior_ind, prior, 
                     misclass_rate, misclass_mat, 
                     return_classes, test_classes, 
                     &one, &zero, (long *) NULL, /* <- xval stuff */
                     increase, &number_of_classes, &verbose);
    }

    *in_best_k = (Slong) best_k;

    if (do_nn_result == FALSE)
    {
        Rprintf ("Panic! do_nn (2) returned FALSE!\n");
        return;
    }

    if (verbose > 1)
        Rprintf ("C  nns %ld, dims %ld, rate is %f\n",
                      k[smallest_misclass_k], 
                      smallest_misclass_dim + 1, misclass_rate[0]);

/* Now call with "quit" set to TRUE to free up memory */
    return_classes = &Slong_FALSE;
    do_nn (&Slong_TRUE, training, test, c, &(k[smallest_misclass_k]), &one, 
        &Slong_FALSE, phi->data, cats_in_var, cum_cats_ptr, knots, cost, 
        &prior_ind, prior, misclass_rate, misclass_mat, FALSE, (Slong *) NULL,
        (long *) NULL, (long *) NULL, (long *) NULL,
        (Slong *) NULL, (Slong *) NULL, (Slong *) NULL);

    free (knots);

} /* end "if CLASSIFICATION" */
else
{
/***
    divide_by_root_prior (original_eigenvectors, prior);
***/
/* 
** "Smallest_misclass_dim" is the dimension of the model, zero-based,
** so it can be zero. If the (one-based) dimension is bigger than
** 1, we have to do our magic with "holes."
*/
    if (smallest_misclass_dim > 0)
    {
        alloc_some_longs (&holes, number_of_vars);
        for (i  = 0; i < smallest_misclass_dim; i++)
            holes[i] = 0L;
        first_time_through = TRUE;
        for (var_ctr = 0; var_ctr < number_of_vars; var_ctr++)
        {
            if (am_i_in[var_ctr] >= CURRENTLY_IN)
            {
                if (first_time_through)
                    first_time_through = FALSE;
                else
                    holes[var_ctr] = cum_cats_this_subset[var_ctr];
            }
        }
    }

    alloc_some_doubles (&misclass_rate, (long) 1);

    do_discriminant (test,
              original_eigenvalues, original_eigenvectors,
              original_w_inv_m_mat, cats_in_var,
              cum_cats_this_subset, 
              (smallest_misclass_dim + 1), number_of_vars, knots, cost, prior,
              misclass_rate, misclass_mat, do_the_omission);	

    Rprintf ("D nns %i, dims %i, rate is %f\n",
                      0,
                      (int) smallest_misclass_dim + 1, misclass_rate[0]);

/**
    Rprintf ("Discrimination misclass_rate is %f\n", *misclass_rate);
**/
}

} /* end "ord" */



/*=========================== matrix_element_divide =======================*/
void matrix_element_divide (MATRIX *result, MATRIX *num, MATRIX *denom)
{
long i, how_many;

how_many = result->nrow * result->ncol;

for (i = 0L; i < how_many; i++)
    if (denom->data[i] < 1e-9)
        result->data[i] = num->data[i] / denom->data[i];
    else
        result->data[i] = 0.0;
}

/*============================== fill_margin_holder ======================*/
int fill_margin_holder (MATRIX *margin_holder, long *permute_index, 
                    long which,
                    MATRIX *data, Slong *cats_in_var, 
                    double **knots, Slong *cum_cats_this_subset, 
                    long *number_in_class, Slong *increase)
/*
** Here we fill up the "margin_holder" matrix. For a categorical
** variable, this holds what you expect; the entry in the i-th row associated 
** with the k-th category of the j-th variable holds the number of observations
** that are in the i-th class and have that category for that variable.
**
** For a continuous variable, it's a little weirder; that entry would have
** the sum of the positive parts of [the x-value for the j-th category
** minus the k-th knot], summed over all observations.
**
** S+ filled up "number_in_class," but we do it again, because we
** need to exclude the items not in this cross-val run.
*/
{
long i, j, knot_ctr;
double this_obs, current_double;
long rows, number_of_doubles;
long this_class, this_col;
long most_recent_non_zero;
int we_ve_seen_the_first_one;

/* Set up a few handy data items. */
rows = data->nrow;
number_of_vars = data->ncol - 1;
number_of_doubles = margin_holder->nrow * margin_holder->ncol;

/*
** Zero out "margin holder," and if we were passed a "number_in_class,"
** zero that out, too.
*/
for (j = 0; j < number_of_doubles; j++)
    margin_holder->data[j] = 0.0;

if (number_in_class != NULL)
    for (j = 0; j < margin_holder->nrow - 1; j++)
        number_in_class[j] = (long) 0;


/*
** Go through this subset of variables, and count up the number of
** categories. "Cum_cats_this_subset[j]" holds the number of categories
** up to **but not including** the j-th variable. If this is a
** permutation, though, we don't have to do this.
*/
we_ve_seen_the_first_one = FALSE;
most_recent_non_zero = 0L;

if (which == DONT_PERMUTE || which == JUST_GET_CUM_CATS)
{
/* Zero out "cum_cats_this_subset" to start. */
    for (j = 0; j < number_of_vars; j++)
        cum_cats_this_subset[j] = 0L;

    for (j = 0; j < number_of_vars; j++)
    {
/* 
** If this variable isn't in, put the last value of "cum_cats_this_subset"
** into the one for this variable, and move on to the next variable.
*/
        if (am_i_in[j] < CURRENTLY_IN)
        {
            if (j > 0)
                cum_cats_this_subset[j] = cum_cats_this_subset[j-1];
            continue;
        }
/* 
** If we've never encountered one a variable that's in the subset
** (which could happen either if we get here when j is 0, or when
** we get here and j isn't zero but "we_ve_seen_the_first_one" is FALSE),
** save the "most_recent_non_zero" (if there's more than one variable), 
** turn on the "we_ve_seen_the_first_one," and then quit. We quit
** because we don't count the number of categories in the j-th variable
** until the j+1-th entry of cum_cats_this_subset. So we need to save
** that number, and that's where "most_recent_non_zero" comes in.
*/
        if (j == 0 || we_ve_seen_the_first_one == FALSE)
        {
            we_ve_seen_the_first_one = TRUE;
            if (number_of_vars > 1)
            {
                most_recent_non_zero = cats_in_var[j];
            }
            continue;
        }
        cum_cats_this_subset[j] = most_recent_non_zero;
        most_recent_non_zero += cats_in_var[j];
    }
}

if (which == JUST_GET_CUM_CATS)
    return TRUE;

/*
** Now we go through each row. We get its class and increment 
** "number_in_class" if we've been asked to.
*/
for (i = 0; i < rows; i++)
{
/*
** Exclude items in the current test set. 
*/
    if (xval_indices != (long *) NULL)
        if (i >= xval_lower && i < xval_upper)
            continue;
    this_class = *SUB (data, i, 0);

    if (number_in_class != (long *) NULL)
        number_in_class[this_class]++;


/*
** Now go through the variables. For each categorical variable,
** find the category for this observation on this variable, and
** add 1 to the appropriate entry in "margin_holder". For each
** continuous variable, add the observation minus the knot
** corresponding to this category, if that's positive, or 0.
*/

    for (j = 1; j <= number_of_vars; j++)
    {
        if (am_i_in[j-1] < CURRENTLY_IN)
            continue;

        if (which == j-1)
            this_obs = *SUB (data, permute_index[i], j);
        else
            this_obs = *SUB (data, i, j);

        if (increase[j-1] == NUMERIC)
        {
/*
** For a numeric, go through the knots for this variable. For each
** knot, add to the entry of margin_holder corresponding to this class
** and knot, the observation minus the knot, if that's positive. The
** knots are in increasing order, so once that diffference is
** negative, we can break.
*/
            for (knot_ctr = 0; knot_ctr < cats_in_var[j-1]; knot_ctr++)
            {
                if (this_obs - knots[j-1][knot_ctr] < 0)
                    break;
                this_col = cum_cats_this_subset[j-1] + knot_ctr;
                current_double = *SUB (margin_holder, this_class, this_col);
                *SUB (margin_holder, this_class, this_col) 
                    = current_double + (this_obs - knots[j-1][knot_ctr]);
                current_double = *SUB (margin_holder, 
                                       number_of_classes, this_col);
                *SUB (margin_holder, number_of_classes, this_col) 
                    = current_double + (this_obs - knots[j-1][knot_ctr]);
            }
        }
        else
        {
/*
** Find the relevant column. It's the number of "cum_cats_this_subset"
** for all the variables up to this one [note that "j" has started at
** 1], plus the categorical associated with this row. ONCE AGAIN WE
** ARE ASSUMING THAT CATEGORIES START AT 0 AND GO UP BY 1'S.
*/
            this_col = cum_cats_this_subset[j-1] + (long) this_obs;
/* Extract the value in the relevant spot in margin_holder and add 1. */
            current_double = *SUB (margin_holder, this_class, this_col);
            *SUB (margin_holder, this_class, this_col) = current_double + 1;

/* Do that for the last row, the totals, as well. */
            current_double = *SUB (margin_holder, number_of_classes, this_col);
            *SUB (margin_holder, number_of_classes, this_col) 
                           = current_double + 1;
        }
    } /* end "j" looping over variables. */

} /* end "i" looping over "rows." */

return TRUE;
} /* end "fill_margin_holder" */

/*============================ fill_u ===================================*/
int fill_u (MATRIX *training, Slong *cats_in_var, Slong *cum_cats_this_subset,
        long *number_in_class, MATRIX *margin_ptr)
{
long cat_row_ctr;

for (cat_row_ctr = 0; cat_row_ctr < number_of_vars; cat_row_ctr ++)
{
    if (am_i_in[cat_row_ctr] < CURRENTLY_IN)
        continue;
    fill_row_of_u_submatrices (training,
         cat_row_ctr, cats_in_var, cum_cats_this_subset,
         number_in_class, margin_ptr);
}

return (TRUE);
} /* end "fill_u" */

/*===================== fill_row_of_u_submatrices ========================*/

int fill_row_of_u_submatrices (MATRIX *training,
    long starting_row, Slong *cats_in_var,
    Slong *cum_cats_this_subset, long *number_in_class, MATRIX *margin_ptr)
{
long row_ctr, big_col_ctr, col_ctr;
long col_start, row_start;
long class_ctr;

double temp;

row_start = cum_cats_this_subset[starting_row];

for (row_ctr = row_start; row_ctr < (row_start + cats_in_var[starting_row]);
     row_ctr++)
{
    for (big_col_ctr = starting_row; big_col_ctr < number_of_vars; 
         big_col_ctr++)
    {
        if (am_i_in[big_col_ctr] < CURRENTLY_IN)
            continue;
        col_start = cum_cats_this_subset[big_col_ctr];
        for (col_ctr = col_start; 
             col_ctr < (col_start + cats_in_var[big_col_ctr]);
             col_ctr++)
        {
            temp = 0.0;
            for (class_ctr = 0L; class_ctr < number_of_classes; class_ctr++)
            {   
                if (number_in_class[class_ctr] == 0)
                {
                    Rprintf ("Number in class %ld is 0! Abort!\n", 
                                     class_ctr);
                    return (FALSE);
                }
                temp += (double) *SUB(margin_ptr, class_ctr, row_ctr) 
                           * (double)*SUB (margin_ptr, class_ctr, col_ctr)
                           / (double) number_in_class[class_ctr];
if (verbose >= 4)
Rprintf ("Row %ld, col %ld, class %ld, adding %f * %f / %ld = %f\n",
row_ctr, col_ctr, class_ctr, 
*SUB(margin_ptr, class_ctr, row_ctr), 
*SUB (margin_ptr, class_ctr, col_ctr), number_in_class[class_ctr], temp);
            }
            *SUB (u, row_ctr, col_ctr) = temp;
            *SUB (u, col_ctr, row_ctr) = temp;
        }
    }
}

return (TRUE);
} /* end "fill_row_of_u_submatrices" */

/*=======================  fill_w ======================================*/

int fill_w (MATRIX *training,
        double **knots, Slong *cats_in_var, Slong *cum_cats_this_subset, 
        MATRIX *margin_ptr, long i, long *permute_indices, Slong *increase)
{
long cat_row_ctr;

for (cat_row_ctr = 0; cat_row_ctr < number_of_vars; cat_row_ctr ++)
{
    if (am_i_in[cat_row_ctr] < CURRENTLY_IN)
        continue;
    fill_row_of_w_submatrices (training,
         cat_row_ctr, knots, cats_in_var, cum_cats_this_subset,
         margin_ptr, i, permute_indices, increase);
}

return (TRUE);
} /* end "fill_w" */

/*===================== fill_row_of_w_submatrices ========================*/
int fill_row_of_w_submatrices (MATRIX *training,
    long starting_row, double **knots, Slong *cats_in_var,
    Slong *cum_cats_this_subset, MATRIX *margin_ptr, long which_to_permute,
    long *permute_indices, Slong *increase)
{
long j, train_n;
long col_ctr, knot_row_ctr, knot_col_ctr;
long this_row, this_col;
long col_start, row_start;

long special_case;
long diag_ctr, diagonal_start, diagonal_finish;

double obs_minus_row_knot, obs_minus_col_knot;
double obs_row, obs_col;
double temp;

train_n = training->nrow;
/*
** Go through the whole training matrix. For each entry go through the
** sub-matrix in this column, and the other sub-matrices to its
** right. If the variables for the current row and the current column 
** are both sub-matrix are:
** (1) both categorical:  then for each observation 
** add 1 to the entry of the current sub-matrix whose row (zero-based)
** is given by the value of the observation for the variable corresonding
** to the current row sub-matrix, and whose column (zero-based) is given 
** by the value of the observation for the variable corresponding to the 
** current column sub-matrix. SPECIAL CASE: if the row and column sub-matrices 
** refer to the same variable, then this is diagonal, with the k-th
** entry giving the number of elements with "k" for this variable.
** (2) row sub-matrix categorical, column sub-matrix numeric:  
** For each observation go to the row of the row sub-matrix that
** corresponds to the value of the observation on the row sub-matrix's
** variable. To the elements of that row add the positive part of [the 
** value of that observation minus the k-th knot], where k goes over the
** knots on the variable corresponding to the column sub-matrix, and
** therefore over the columns of the column sub-matrix.
** (3) row sub-matrix numeric, column sub-matrix categorical:  
** Same as above, with rows and columns interchanged.
** (4) Double-loop over the knots associated with the variable
** corresponding to the row sub-matrix and the one corresponding to
** the column sub-matrix. For each knot combination, add to the
** corresponding entry in M the product of pos.part (obs.on row var. - row knot)
** and pos.part (obs. on column var. - column knot.)
*/

row_start = cum_cats_this_subset[starting_row];

if (increase[starting_row] == NUMERIC)
{
    for (j = 0L; j < train_n; j++)
    {
        if (j >= xval_lower && j < xval_upper)
            continue;
        for (col_ctr = starting_row; col_ctr < number_of_vars; col_ctr++)
        {
            if (am_i_in[col_ctr] < CURRENTLY_IN)
                continue;
            col_start = cum_cats_this_subset[col_ctr];
            obs_row = (starting_row == which_to_permute ? 
                *SUB (training, permute_indices[j], starting_row + 1) :
                *SUB (training, j, starting_row + 1));
            if (increase[col_ctr] == NUMERIC)
            {
                for (knot_row_ctr = 0; knot_row_ctr < cats_in_var[starting_row];
                     knot_row_ctr++)
                {
                    obs_minus_row_knot = 
		        obs_row - knots[starting_row][knot_row_ctr];
                    if (obs_minus_row_knot < 0)
                        break;
                    obs_col = (col_ctr == which_to_permute ? 
                        *SUB (training, permute_indices[j], col_ctr + 1) :
                        *SUB (training, j, col_ctr + 1));
                    for (knot_col_ctr = 0; knot_col_ctr < cats_in_var[col_ctr]; 
                         knot_col_ctr++)
                    {
                        obs_minus_col_knot = 
			    obs_col - knots[col_ctr][knot_col_ctr];
if (verbose > 4)
Rprintf ("Obs %ld is %f, minus knot %ld %ld -- %f -- gives %f\n",
j,  *SUB (training, j, col_ctr + 1), col_ctr, knot_col_ctr,
knots[col_ctr][knot_col_ctr],
obs_minus_col_knot);

                        if (obs_minus_col_knot < 0)
                            break;
                        temp = *SUB (w, knot_row_ctr + row_start, 
                                     knot_col_ctr + col_start);
                        *SUB (w, knot_row_ctr + row_start, 
                              knot_col_ctr + col_start) = 
                            temp + obs_minus_row_knot * obs_minus_col_knot;

if (verbose > 4)
/* if (knot_row_ctr + row_start == 0 && knot_col_ctr + col_start == 0) */
Rprintf ("%ld: Adding %f * %f = %f to spot %ld %ld, giving %f\n", j,
obs_minus_row_knot, obs_minus_col_knot,
obs_minus_row_knot * obs_minus_col_knot,
knot_row_ctr + row_start, knot_col_ctr + col_start,
temp + obs_minus_row_knot * obs_minus_col_knot);
                        if (starting_row != col_ctr)
                            *SUB (w, knot_col_ctr + col_start, 
                                  knot_row_ctr + row_start) = 
                                temp + obs_minus_row_knot * obs_minus_col_knot;
                    } /* for "knot_col_ctr" loop */
                } /* end "for knot_row_ctr" loop */

            } /* end "if col is numeric" */
            else
            {    
/* Obs row is alreasdy set. */
                this_col =  cum_cats_this_subset[col_ctr] +
/*****                      cum_cats_this_subset[col_ctr-1]; *****/
			    (col_ctr == which_to_permute ?
		                (long) *SUB (training, permute_indices[j], 
					     col_ctr + 1) :
		                (long) *SUB (training, j, col_ctr + 1));
                for (knot_row_ctr = 0; knot_row_ctr < cats_in_var[starting_row];
                         knot_row_ctr++)
                {
                    obs_minus_row_knot = 
		        obs_row - knots[starting_row][knot_row_ctr];
                    if (obs_minus_row_knot < 0)
                        break;
                    temp = *SUB (w, knot_row_ctr + row_start, this_col);
                    *SUB (w, knot_row_ctr + row_start, this_col) =
                        temp + obs_minus_row_knot;
                    *SUB (w, this_col, knot_row_ctr + row_start) =
                        temp + obs_minus_row_knot;
                }
            } /* end "else," i.e. if col is categorical */
        } /* end "for col_ctr" loop */
    } /* end "for j" loop. */
} /* end "if this row sub-matrix's variable is numeric" */
else
{
/* Special case: for the diagonal submatrices, we don't have to count through
** the whole file. The number in the k-th slot is just the number of
** observations with a "k" there, and we have that information in
** the last row of "margin_ptr".
*/
    if (increase[starting_row] == NUMERIC)
        special_case = 0;
    else
    {
        special_case = 1;
        diagonal_start = cum_cats_this_subset[starting_row];
        diagonal_finish = diagonal_start + cats_in_var[starting_row];
        for (diag_ctr = diagonal_start; diag_ctr < diagonal_finish; diag_ctr++)
        {
            *SUB (w, diag_ctr, diag_ctr) = 
                  *SUB (margin_ptr, number_of_classes, diag_ctr);
        }
    }


    for (j = 0L; j < train_n; j++)
    {
        if (j >= xval_lower && j < xval_upper)
            continue;
        this_row = row_start + (starting_row == which_to_permute ?
		   (long) *SUB (training, permute_indices[j], starting_row + 1):
		   (long) *SUB (training, j, starting_row + 1));
        for (col_ctr = starting_row + special_case; col_ctr < number_of_vars; 
             col_ctr++)
        {
            if (am_i_in[col_ctr] < CURRENTLY_IN)
                continue;
            col_start = cum_cats_this_subset[col_ctr];
            if (increase[col_ctr] == NUMERIC)
            {
		obs_col = (col_ctr == which_to_permute?
		              *SUB (training, permute_indices[j], col_ctr + 1):
		              *SUB (training, j, col_ctr + 1));
                for (knot_col_ctr = 0; knot_col_ctr < cats_in_var[col_ctr]; 
                     knot_col_ctr++)
                {
                    obs_minus_col_knot = obs_col - knots[col_ctr][knot_col_ctr];
                    if (obs_minus_col_knot < 0)
                        break;
                    temp = *SUB (w, this_row, knot_col_ctr + col_start);
                    *SUB (w, this_row, knot_col_ctr + col_start) = 
                        temp + obs_minus_col_knot;
                    *SUB (w, knot_col_ctr + col_start, this_row) =
                        temp + obs_minus_col_knot;

if (verbose > 4)
Rprintf("%ld: Adding obs - knot %ld (%f) to spot %ld %ld giving %f\n",
j, knot_col_ctr, obs_minus_col_knot, this_row, knot_col_ctr + col_start,
temp + obs_minus_col_knot);

                }
            }
            else
            {
                this_col = cum_cats_this_subset[col_ctr] + 
                           (col_ctr == which_to_permute?
                               (long) *SUB (training, permute_indices[j], 
					    col_ctr + 1):
                               (long) *SUB (training, j, col_ctr + 1));
/******
Rprintf ("Cat col is (cum) %ld + (entry) %ld (choices %ld and %ld), = %ld\n", 
cum_cats_this_subset[col_ctr], 
col_ctr == which_to_permute?
(long) *SUB (training, permute_indices[j], col_ctr + 1):
(long) *SUB (training, j, col_ctr + 1),
(long) *SUB (training, permute_indices[j], col_ctr + 1),
(long) *SUB (training, j, col_ctr + 1),
this_col);
******/
/****                  + cum_cats_this_subset[col_ctr-1]; ****/
                temp = *SUB (w, this_row, this_col) + 1.0;
                *SUB (w, this_row, this_col) = temp;
                *SUB (w, this_col, this_row) = temp;
            } /* end "else" on "if col is numeric" */
        } /* end "for col_ctr" loop */
    } /* end "for j" loop */
} /* end "else" on "if row sub-matrix is numeric" */

return (TRUE);

} /* end "fill_row_of_w_submatrices" */


/*================= divide_by_root_prior ====================*/

int divide_by_root_prior (MATRIX *in, MATRIX *prior)
{
/*
** There's a trick here. The "in" matrix has been transposed. So
** we divide by the entry in the column in stead of the row.
*/
unsigned long i, j;
double jth_diag;

if (prior == (MATRIX *) NULL)
    return (TRUE);

for (j = 0; j < in->ncol; j++)
{
    jth_diag = *SUB (prior, j, j);
    if (jth_diag == 0.0)
    {
        Rprintf ("Abort! %ldth Prior is 0!\n", j);
        return (FALSE);
    }
    for (i = 0; i < in->nrow; i++)
        *SUB (in, i, j) = *SUB (in, i, j) / sqrt (jth_diag);
}

return (TRUE);
}

/*================= multiply_by_root_prior ====================*/

int multiply_by_root_prior (MATRIX *in, MATRIX *prior)
{
unsigned long i, j;
double jth_diag;

if (prior == (MATRIX *) NULL)
    return (TRUE);

if (in->ncol != prior->ncol)
{
    Rprintf ("Big time trouble.\n");
    return (NON_CONFORMABLE);
}

for (j = 0; j < in->ncol; j++)
{
    jth_diag = *SUB (prior, j, j);
    for (i = 0; i < in->nrow; i++)
        *SUB (in, i, j) = *SUB (in, i, j) * sqrt (jth_diag);
}

return (TRUE);
}

/*=========================== resize_matrix ==============================*/
int resize_matrix (MATRIX **a, long nrow, long ncol)
{
/*
** Resize (or create) a to be nrow by ncol.
*/ 
/* char *realloc_result; */
unsigned long how_many;

if (*a == (MATRIX *) NULL)
{
    *a = make_matrix (nrow, ncol, "No Name", REGULAR, TRUE);
}
else
{
    if ((*a)->nrow != nrow || (*a)->ncol != ncol)
    {
        (*a)->nrow = nrow;
        (*a)->ncol = ncol;
        how_many = nrow * ncol;
        if ((*a)->data == (double *) NULL)
            alloc_some_doubles (&((*a)->data), how_many);
        else
        {
            free ((*a)->data);
            alloc_some_doubles (&((*a)->data), how_many);
        }
/***
        realloc_result = realloc ((*a)->data, 
                             (*a)->nrow * (*a)->ncol * sizeof (double));
        if (realloc_result == (char *) NULL)
        {
            Rprintf ("Resize matrix: realloc failed\n");
            return (FALSE);
        }
***/
    }
}

return (TRUE);
} /* end "resize_matrix" */

/*=========================== adjust_column ==============================*/
int adjust_column (unsigned long column, MATRIX *adjustee, long increase,
                   double **knots, long cats_in_this_var, MATRIX *phi)
{
/*
** This replaces the ith element in column "column" of the adjustee 
** matrix by its new value. For a numeric column, this value is the
** sum of (old element - knot) for all the knots where this is
** positive. For a categorical, we look at the ith element for this
** variable, which we find in the ith spot after the first spot
** for this variable, which latter is in cum_cats_this_subset[column].
*/

unsigned long nrow;
unsigned long j, knot_ctr;
unsigned long offset;
double temp;
double obs_minus_knot;

nrow = adjustee->nrow;
offset = cum_cats_this_subset[column];

for (j = 0; j < nrow; j++) 
{
    if (increase == NUMERIC)
    {
        temp = 0.0;
        for (knot_ctr = 0; knot_ctr < cats_in_this_var; knot_ctr++)
        {
            obs_minus_knot = *SUB (adjustee, j, column + 1) - 
                                 knots[column][knot_ctr];
            if (obs_minus_knot <= 0.0)
                break;
            temp += obs_minus_knot;
/**
* 
                phi->data[knot_ctr + offset];
**/
            if (verbose > 1)
Rprintf ("Obs %ld: # %f - knot (%ld) %f  makes temp %f\n",
j, *SUB (adjustee, j, column + 1), knot_ctr, obs_minus_knot, temp);
        }
        *SUB (adjustee, j, column + 1) = temp;
    }
    else
        *SUB (adjustee, j, column + 1) 
            = phi->data[(long) *SUB (adjustee, j, column + 1) + offset];

} /* end "for j" looping over matrix */
 
return (TRUE);
} /* end "adjust_column" */

/*====================== add_ordered_variable =========================*/
double add_ordered_variable (double *result, long which_var, 
                      long dimension, Slong *cats_in_var, MATRIX *phi,
                      Slong *increase)
{
/*
** Each ordered variable requires (# of cats) lin. constraints. (# of cats - 1)
** are of the form c_i - c_(i+1) > 0 (for increasing, or < 0 for decreasing).
** The final one is that the weighted mean of the parameters for this
** variable be 0. Unordered variables just get this last one.
*/
long cat_ctr, cat_start;
long j, j_start, both_are_bad;
long var_ctr, increase_ctr;
long temp_long;

/* long e04_iter = 350L; */
long e04_msglvl = 1L;
long lin_const_count, nonlin_const_count = 1;
double nonlin_const_value;
long nc_total;
/**
double e04_tolerance = 1e-7;
double e04_tolerance2 = 1e-7;
**/
double *lower_bounds, *upper_bounds;
long *istate;
double *jacobian_result;
double objective_result = 0.0, save_objective_result = 0.0;
double *gradient_result;
double *e04_lambda;
long *long_work; long length_long_work;
double *double_work; long length_double_work;
MATRIX *lin_const_matrix;
MATRIX *save_phi;
long orderable_cats, currently_ordered, current_cat_total;
long constraint_one = 1L;
int adjust_this_one;
static long last_variable_ordered = -2;

/* Variables specific to e04vcf */
/* double bigbnd, epsaf, eta, *featol; */
/* int cold, fealin, orthog; */
/* MATRIX *R; */
/*=============================*/
int set_WSS_to_1_in_a_desperate_hope_to_get_the_constraint_working = FALSE;
int christ_im_desperate = FALSE;

if (increase[which_var] == UNORDERED || increase[which_var] == NUMERIC)
    adjust_this_one = FALSE;
else
{
    adjust_this_one = TRUE;
    last_variable_ordered = which_var;
}
current_cat_total = 0L;
currently_ordered = 0L;
orderable_cats    = 0L;
for (var_ctr = 0; var_ctr < number_of_vars; var_ctr++)
{
    if (am_i_in[var_ctr] >= CURRENTLY_IN) {
        current_cat_total += cats_in_var[var_ctr];
        if (increase[var_ctr] != UNORDERED
        &&  increase[var_ctr] != NUMERIC)
        {
            currently_ordered++;
            orderable_cats += cats_in_var[var_ctr];
        }
    }
}

lin_const_count = orderable_cats + (dimension - currently_ordered);
        
nc_total = current_cat_total + lin_const_count + nonlin_const_count;

alloc_some_doubles (&lower_bounds, nc_total);
alloc_some_doubles (&upper_bounds, nc_total);
/* phi = make_matrix (1L, current_cat_total, "Phi", REGULAR, TRUE); */

alloc_some_longs   (&istate, nc_total);
alloc_some_doubles (&jacobian_result, current_cat_total);
alloc_some_doubles (&gradient_result, current_cat_total);
alloc_some_doubles (&e04_lambda, nc_total);

length_long_work = 3 * current_cat_total + lin_const_count + nonlin_const_count;
alloc_some_longs (&long_work, length_long_work);

length_double_work  = 3 * current_cat_total * current_cat_total
                + current_cat_total * (lin_const_count + 19)
                + lin_const_count * 8 + 9 + 1000;
alloc_some_doubles (&double_work, length_double_work);

save_phi = make_matrix (phi->nrow, phi->ncol, "Save Phi", REGULAR, TRUE);
matrix_copy (save_phi, phi);

/* Set the bounds on the parameters. The constraints come later. */
for (cat_ctr = 0; cat_ctr < current_cat_total; cat_ctr++)
{
    lower_bounds[cat_ctr] = -1.0e+21;
    upper_bounds[cat_ctr] =  1.0e+21;
}

/* The non-linear constraint. */
lower_bounds[current_cat_total + lin_const_count] = 1.0;
upper_bounds[current_cat_total + lin_const_count] = 1.0;

lin_const_matrix = make_matrix (current_cat_total, lin_const_count,
                        "Lin Const", REGULAR, TRUE);
/*
** Preparations specific to e04vcf
**
**
** R = make_matrix (current_cat_total, current_cat_total, "R", REGULAR, TRUE);
** alloc_some_doubles (&featol, nc_total);
*/
/*
** Okay. If we don;'t know which way "which_var" (the one being added) 
** goes, go through this bit twice, once for increasing and once for 
** decreasing. But we come here after *any* variable is added, once there's
** at least one ordered variable in the model. So it could happen that
** "which_var" is numeric or unordered. In that case we need go through this
** only once.
*/
both_are_bad = 0;
/* This line relies on #defines of DECREASING = 0 and INCREASING = 1. */

if (adjust_this_one)
    increase_ctr = 0L;
else
    increase_ctr = 1L;
for (; increase_ctr <= 1L; increase_ctr++)
{
    zero_matrix (lin_const_matrix);
    if (adjust_this_one)
        increase[which_var] = increase_ctr;  /* Relies on **CREASING defines */

/*** Old way of setting up initial phi values. Wrong, wrong, wrong.
HERE! HERE! HERE!
    if (increase[which_var] == DECREASING)
        for (j = 0; j < current_cat_total; j++)
            phi->data[j] = (current_cat_total - 2 * j) / current_cat_total;
    else
        for (j = 0; j < current_cat_total; j++)
            phi->data[j] = - (current_cat_total - 2 * j) / current_cat_total;
***/
           
    if (increase_ctr == 1)
        matrix_copy (phi, save_phi);

    temp_long = 0L;    /* Which constraint are we on?    */
    j_start = 0L;      /* First column for this variable */

    for (var_ctr = 0; var_ctr < number_of_vars; var_ctr++)
    {
        if (am_i_in[var_ctr] < CURRENTLY_IN)
        {
            if (verbose > 1)
                Rprintf ("Variable %ld? Not in.\n", var_ctr);
            continue;
        }

        cat_start = cum_cats_this_subset[var_ctr];
        if (verbose > 1)
            Rprintf ("Var ctr %ld, cat_start %ld\n", var_ctr, cat_start);

/* "j" counts through the columns of this constraint. It starts
** at 0, but we operate starting wherever "j_start" starts.
**
** The first constraint is the weighted mean = 0 one. We take the
** diagonal entries of W for this variable, and require that the sum of
** those numbers times the corresponding coefficients be 0.
*/
        for (j = 0; j < cats_in_var[var_ctr]; j++)
        {
            *RSUB (lin_const_matrix, temp_long, j_start + j) 
                = *SUB (global_copy_of_w, cat_start + j, cat_start + j);
             if (verbose > 1)
                 Rprintf ("Loadin' from w %ld %ld into const  %ld %ld\n", 
                     cat_start + j, cat_start + j, temp_long, j_start + j);
        }
        lower_bounds[current_cat_total + temp_long] =  0.0;
        upper_bounds[current_cat_total + temp_long] =  0.0;
        temp_long++;

        if (increase[var_ctr] == UNORDERED 
        ||  increase[var_ctr] == NUMERIC)
        {
            j_start += cats_in_var[var_ctr];
            if (verbose > 1)
                Rprintf ("That's all for variable %ld\n", var_ctr);
            continue;
        }

        for (j = j_start; j < j_start + cats_in_var[var_ctr] - 1; j++)
        {
if (verbose > 1)
{
    Rprintf ("Putting %ld into const %ld %ld, other into %ld %ld...\n",
    (long) (increase[var_ctr] == DECREASING ? 1L : (long) -1), temp_long, j,
    temp_long, j+1);
}
            *RSUB (lin_const_matrix, temp_long, j)
                = (increase[var_ctr] == DECREASING ? 1.0 : -1.0);
            *RSUB (lin_const_matrix, temp_long, j + 1)
                = (increase[var_ctr] == DECREASING ? -1.0 : 1.0);
            lower_bounds[current_cat_total + temp_long] =  0.0;
            upper_bounds[current_cat_total + temp_long] =  1.0e+21;
	    temp_long++;
        }
        j_start += cats_in_var[var_ctr];
    } /* end "for var_ctr" */

/************* Global "weighted mean = 0" constraint ***************
********    for (j = 0; j < current_cat_total; j++)
********    {
********        *RSUB (lin_const_matrix, lin_const_count - 1, j) 
********            = *SUB (w, j, j);
********    }
********/

if (set_WSS_to_1_in_a_desperate_hope_to_get_the_constraint_working)
{
/* double duty: e04_msglvl is not used in constraint. */
    constraint (&e04_msglvl, &nonlin_const_count, &current_cat_total,
    &nonlin_const_count, phi->data, &nonlin_const_value,
    jacobian_result, &constraint_one);
    for (j = 0; j < current_cat_total; j++)
        phi->data[j] /= sqrt (nonlin_const_value);
}
    NAG_status = 0L;

/***
    e04zcf_ (&current_cat_total, 
             &nonlin_const_count,
             &nonlin_const_count,
             &constraint, &objective, 
             &nonlin_const_value, jacobian_result, &objective_result,
             gradient_result, phi->data,
             double_work, &length_double_work, &NAG_status);

Rprintf ("Status from e04zcf is %ld\n", NAG_status);
    NAG_status = 1L;
    e04_iter = (long) 350;
    e04_msglvl = 1L;

    e04vdf_ (&e04_iter, &e04_msglvl, &current_cat_total, 
             &lin_const_count, &nonlin_const_count, &nc_total, 
             &lin_const_count, &nonlin_const_count, &e04_tolerance, 
             &e04_tolerance2, 
             lin_const_matrix->data, lower_bounds, upper_bounds,
             &constraint, &objective, phi->data, istate, 
             &nonlin_const_value, jacobian_result, &objective_result,
             gradient_result, e04_lambda, long_work, &length_long_work, 
             double_work, &length_double_work, &NAG_status);
**/

/*
** Variables specific to e04vcf 
*/

/*
     bigbnd = 1.0e+20;
     epsaf  = 1.0e-06;
     eta    = 0.9;
     for (j = 0; j < nc_total; j++)
         featol[j] = 1.0e-7;
     cold   = TRUE;
     fealin = TRUE;
     orthog = TRUE;
     e04_msglvl = 0L;
     e04_iter   = 30L;
*/
     NAG_status = 1;
     christ_im_desperate = TRUE;

     while (christ_im_desperate == TRUE)
     {
#if 0
     e04vcf_ (&e04_iter, &e04_msglvl, &current_cat_total, 
              &lin_const_count, &nonlin_const_count,
              &nc_total, &lin_const_count, &nonlin_const_count,
              &current_cat_total, &bigbnd, &epsaf, &eta,
              &e04_tolerance, lin_const_matrix->data,
              lower_bounds, upper_bounds, 
              featol, &constraint, &objective,
              &cold, &fealin, &orthog,
              phi->data, istate, 
              R->data, &iter_result, 
              &nonlin_const_value, jacobian_result,
              &objective_result, gradient_result,
              e04_lambda, long_work, &length_long_work,
              double_work, &length_double_work, &NAG_status);
#endif

    if (NAG_status == 0)
        break;
    if (NAG_status == 3)
    {
/*
        change_to_the_identity (R);
        cold = FALSE;
*/
        Rprintf ("Stuck in the NAG status = 3 loop!\n");
        continue;
    }
    break;
    } /* end "while" */

    if (increase_ctr == 0)
        save_objective_result = - objective_result;

    if (NAG_status != 0)
    {
        Rprintf ("Serious trouble: NAG_status is %ld\n", NAG_status);
        if (NAG_status == 2L)
        {
            Rprintf ("Warning: one non-lin gave NAG 2\n");
        }
        else
        {
            save_objective_result = - 1.0;
            both_are_bad++;
        }
    }

} /* end "for increase_ctr" */


if (both_are_bad == 2 || (both_are_bad == 1 && !adjust_this_one))
{
/* Leave "result" unchanged */
    Rprintf ("Huge Trouble: Both Failed!\n");
    matrix_copy (phi, save_phi);
    if (adjust_this_one)
    {
        increase[which_var] = UNORDERED;
        Rprintf ("Number %ld is now unordered!\n", which_var);
    }
    else
    {
        Rprintf ("That's weird; we're on non-adjusted number %ld,",
                 which_var);
        Rprintf (" unordering last ordered number %ld\n",
                last_variable_ordered);
        increase[last_variable_ordered] = UNORDERED;
    }
        
}
else
{
/* Remember the objective is -TSS, so make it positive. */
    *result = - objective_result;
    if (save_objective_result < objective_result)
    {
        if (adjust_this_one)
        {
            increase[which_var] = INCREASING;
            Rprintf ("ld is increasing\n", which_var);
        }
    }
    else
    {
        *result = save_objective_result;
        objective_result = save_objective_result;
        if (adjust_this_one)
        {
            increase[which_var] = DECREASING;
            Rprintf ("ld is decreasing\n", which_var);
        }
    }
}

free (lower_bounds);
free (upper_bounds);
free (istate);
free (jacobian_result);
free (gradient_result);
free (e04_lambda);
free (long_work);
free (double_work);
free (lin_const_matrix->data);
free (save_phi->data);
/*
** Specific to e04vcf
** free (R->data);
** free (featol);
*/

return (objective_result);

} /* end add_ordered_variable */

/*====================== deal_with_missing_values =========================*/
void deal_with_missing_values (MATRIX *training, double missing_max, 
                              Slong *increase, Slong *cats_in_var, 
                              double *missing_values)
{
/*
** Here's where we decide what missing values ought to be. The deal for
** now is that we'll use the mean for numerics and the most common for
** categoricals. We call "provisional_means" once for each numeric, 
** non-missing variable: it's too much hassle to build a vector each
** time to use the "provisional_means" vector-handling capability,
** because some rows have missing values. For categoricals we set up
** a vector of the proper length (from cats_in_var) and count, count, 
** count.
*/

long i, cat_ctr, col_ctr, row_ctr;
long number_of_vars, number_of_rows;
long this_cat;
long **cat_totals;
double current_mean;
long biggest_cat, biggest_cat_count;
int some_were_missing;

number_of_vars = training->ncol - 1;
alloc_some_long_ptrs (&cat_totals, number_of_vars);

/* Here we start col_ctr at 0. We're counting data things. */
for (col_ctr = 0; col_ctr < number_of_vars; col_ctr++)
{
    alloc_some_longs (&(cat_totals[col_ctr]), cats_in_var[col_ctr]);
    for (i = 0; i < cats_in_var[col_ctr]; i++)
    {
        cat_totals[col_ctr][i] = 0.0;
    }
}

number_of_rows = training->nrow;
for (col_ctr = 0; col_ctr < number_of_vars; col_ctr++)
{
    if (increase[col_ctr] == NUMERIC)
    {
        current_mean = 0.0;
        some_were_missing = FALSE;
        for (row_ctr = 0; row_ctr < number_of_rows; row_ctr ++)
        {
            this_cat = *SUB (training, row_ctr, 0L);
            if (this_cat > missing_max)
            {
                some_were_missing = TRUE;
                provisional_means (SUB (training, row_ctr, col_ctr + 1), 1L,
                                   INCREMENT, &current_mean, (double *) NULL); 
            }
        }
        if (some_were_missing == TRUE)
            missing_values[col_ctr] = current_mean;
        else
            missing_values[col_ctr] = missing_max;
        provisional_means ((double *) NULL, 1L, QUIT, (double *) NULL,
                           (double *) NULL); 
    }
    else
    {
        some_were_missing = FALSE;
        for (row_ctr = 0; row_ctr < number_of_rows; row_ctr ++)
        {
            this_cat = (long) *SUB (training, row_ctr, col_ctr + 1);
            if (this_cat > missing_max)
            {
                some_were_missing = TRUE;
                cat_totals[col_ctr][this_cat]++;
            }
        }
        if (some_were_missing == TRUE)
        {
            biggest_cat = -1;
            biggest_cat_count = -1;
            for (cat_ctr = 0; cat_ctr < cats_in_var[col_ctr]; cat_ctr++)
            {
                if (biggest_cat < 0 
                ||  biggest_cat_count < cat_totals[col_ctr][cat_ctr])
                {
                    biggest_cat = cat_ctr;
                    biggest_cat_count = cat_totals[col_ctr][cat_ctr];
                }
            } /* end "for cat_ctr" */
            missing_values[col_ctr] = biggest_cat;
        }
        else
            missing_values[col_ctr] = missing_max;
    } /* else (that is, if this is categorical) */
} /* end "for col_ctr" */

for (col_ctr = 0; col_ctr < number_of_vars; col_ctr++)
{
    free (cat_totals[col_ctr]);
}

/* free (cat_totals); */

} /* end "deal_with_missing_values" */

/*====================== insert_missing_values =========================*/
int insert_missing_values (MATRIX *mat, double missing_max, 
                           double *missing_values)
{
long col_ctr, row_ctr;
long number_of_rows, number_of_vars;
/*
** Go through the matrix, replacing missing values (things smaller than
** missing_max) with corresponding entries in "missing_values".
*/
number_of_vars = mat->ncol - 1;
number_of_rows = mat->nrow;
for (col_ctr = 0; col_ctr < number_of_vars; col_ctr++)
{
    for (row_ctr = 0; row_ctr < number_of_rows; row_ctr ++)
    {
        if (*SUB (mat, row_ctr, col_ctr + 1) < missing_max)
            *SUB (mat, row_ctr, col_ctr + 1) = missing_values[col_ctr];
    }
}

return (TRUE);

} /* end "insert_missing_values" */

/*====================== do_the_eigen_thing =========================*/
int do_the_eigen_thing (MATRIX *eigenval_ptr, 
                    MATRIX *eigenvec_ptr, double ridge,
                    MATRIX *eigenvalues_imaginary, MATRIX *eigenvalues_beta)
{
long LAPACK_status;
long problem_type = 1L;
double *work_1;
long length_of_work_1;
char the_letter_V = 'V', the_letter_U = 'U';
long i;

if (current_cat_total > number_of_classes)
    length_of_work_1 = 3 * current_cat_total;
else
    length_of_work_1 = 3 * number_of_classes;
alloc_some_doubles (&work_1, length_of_work_1);

matrix_copy (global_copy_of_u, u);
if (ridge > 0.0)
{
    matrix_ridge (w, ridge);
    matrix_copy (global_copy_of_w, w);

F77_CALL(dsygv) (&problem_type, &the_letter_V, &the_letter_U, 
            (long *) &(u->nrow), u->data, (long *) &(u->nrow), 
            w->data, (long *) &(w->nrow), eigenval_ptr->data,
            work_1, &length_of_work_1, &LAPACK_status);
    free (work_1);

    if (LAPACK_status != 0) {
        print_matrix (global_copy_of_u, 8);
        print_matrix (global_copy_of_w, 8);
        return (LAPACK_status);
    }

    eigenvec_ptr->data = u->data;

    if (eigenvec_ptr->nrow == 1)
        eigenvec_ptr->data[0] = 1.0;
}
else
{
    matrix_copy (global_copy_of_w, w);
#ifdef NAG
    alloc_some_longs (&eigen_iterations, current_cat_total);
    f02bjf_(&(u->nrow), u->data, &(u->nrow), w->data, &(w->nrow), 
            &eigen_tolerance, eigenval_ptr->data, 
            eigenvalues_imaginary->data, eigenvalues_beta->data,
            &gimme_the_damn_eigenvectors_or_ill_kill_you_and_i_mean_it,
            eigenvec_ptr->data, &(eigenvec_ptr->nrow), 
            eigen_iterations,
            &NAG_status);
#else
    Rprintf ("Sorry; ridge currently required to be != 0!\n");
    return(-1);
#endif
    for (i = 0; i < eigenval_ptr->nrow; i++)
        eigenval_ptr->data[i] /= eigenvalues_beta->data[i];


/***
    free (eigen_iterations);
    rgg_(&(u->nrow), &(u->nrow), u->data, w->data,
         eigenval_ptr->data, eigenvalues_imaginary->data, 
         eigenvalues_beta->data,
         &gimme_the_damn_eigenvectors_long, eigenvec_ptr->data,
         &NAG_status);
***/
}

if (NAG_status != 0)
    Rprintf ("Eigen NAG status is %li\n", NAG_status);

return (0);
} /* end "do_the_eigen_thing" */

/*====================== get_a_solution =========================*/
int get_a_solution (long permute_ctr, long *permute_indices,
                    long permute_len, long current_var, 
                    MATRIX *margin_ptr, MATRIX *training, 
                    Slong *cats_in_var, Slong *cum_cats_this_subset, 
                    double **knots,
                    long *number_in_class, MATRIX *eigenval_ptr, 
                    MATRIX *eigenvec_ptr, MATRIX *w_inv_m, double local_ridge, 
                    int classification, long number_of_vars, int dimension, 
                    long current_cat_total, MATRIX **prior, Slong prior_ind, 
                    int number_of_classes, int first_time_through, 
                    int do_the_omission, Slong *increase, int quit)
{
/* static MATRIX *u; */
static MATRIX *eigenvalues_imaginary, *eigenvalues_beta;
long i, class_ctr;
int do_eigen_result;
int permute_or_dont;
int we_ve_computed_prior = FALSE;

if (quit)
{
    if (classification == CLASSIFICATION && ridge <= 0.0)
    {
        free (eigenvalues_imaginary->data);
        free (eigenvalues_beta->data);
    }
    return (0);
}

if (permute_ctr == 0)
{
    if (first_time_through == FALSE)
    {
        /* free (u->data); free (w->data);  */
        if (classification == CLASSIFICATION) 
        {
            if (ridge <= 0.0)
            {
                free (eigenvalues_imaginary->data);
                free (eigenvalues_beta->data);
            }
        }
    }
    else
        for (i = 0; i < train_n; i++)
            permute_indices[i] = i;

    if (ridge <= 0.0)
    {
        eigenvalues_imaginary  = make_matrix (current_cat_total, 1L, 
                                     "EigenIm", REGULAR, ZERO_THE_MATRIX);
        eigenvalues_beta       = make_matrix (current_cat_total, 1L, 
                                     "EigenBeta", REGULAR, ZERO_THE_MATRIX);
    }

} /* end "if permute_ctr == 0" */
else
{
    zero_matrix (u);
    zero_matrix (w);
    genprm (permute_indices, (int) permute_len);
}


if (current_var == -1)
    permute_or_dont = -1;
else
    permute_or_dont =  (permute_ctr == 0 ? DONT_PERMUTE : current_var);
fill_margin_holder (margin_ptr, permute_indices, 
                    (long) (permute_or_dont),
                    training, cats_in_var, knots,
                    cum_cats_this_subset, number_in_class, increase);

/*
** Last of the first_time stuff: priors.
*/
if (we_ve_computed_prior == FALSE)
{
/*
** This seems like a weird place, but we've finally found out how many
** observations are in each class. So this is a good place to estimate
** the "prior" probabilities (unless they were supplied, or the user
** asked that they all be equal). Our estimate for the i-th class is, 
** of course, (# in class i)/(# in training set).
**
** We need to do this every time through, since our estimates of the
** prior probabilities will change slightly from xval to xval.
*/
    if (prior_ind == SUPPLIED)
    {
        for (class_ctr = 0; class_ctr < number_of_classes; class_ctr++)
            *SUB ((*prior), class_ctr, class_ctr) = priordata[class_ctr];
    }
    if (prior_ind != SUPPLIED) 
    {
        if (prior_ind == ESTIMATED)
        {
/*** DELETED FOR NOW ***
****        if (*prior != (MATRIX *) NULL)
****            free_matrix (*prior);
****        *prior = make_matrix (number_of_classes, number_of_classes, 
****                             "Prior", REGULAR, ZERO_THE_MATRIX);
***/
            for (class_ctr = 0; class_ctr < number_of_classes; class_ctr++)
            {
                *SUB ((*prior), class_ctr, class_ctr) 
                       = (double) number_in_class[class_ctr] / 
                         (double) train_n_effective;
                priordata[class_ctr] = *SUB ( (*prior), class_ctr, class_ctr);
             }
        }
        else
        {
            if (prior_ind == ALL_EQUAL)
            {
/*** DELETED FOR NOW
****            *prior = make_matrix (number_of_classes, number_of_classes, 
****                                 "Prior", REGULAR, ZERO_THE_MATRIX);
***/
                for (class_ctr = 0; class_ctr < number_of_classes; class_ctr++)
                {
                    *SUB ((*prior), class_ctr, class_ctr) 
                                     = 1.0 / (double) number_of_classes;
                    priordata[class_ctr] = 
                        *SUB ( (*prior), class_ctr, class_ctr);
                }
            }
/* 
** Here's where we would say "else prior ind must be IGNORED", but in
** this case there's nothing to do. Prior is a NULL matrix anyway.
*/
        }
    }
    we_ve_computed_prior = TRUE;
} /* end "if we_ve_computed_prior == FALSE" */


/* This looks weird, but recall that w is globally defined. */
fill_w (training, /* EXPLAIN permute_or_dont */
        knots, cats_in_var, cum_cats_this_subset,
        margin_ptr, permute_or_dont, permute_indices, increase); 

if (classification == CLASSIFICATION)
{
    fill_u (training, cats_in_var, cum_cats_this_subset,
            number_in_class, margin_ptr);
                        
    do_eigen_result = do_the_eigen_thing (eigenval_ptr, eigenvec_ptr, ridge,
                        eigenvalues_imaginary, eigenvalues_beta);

    if (do_eigen_result != 0)
    {
        Rprintf ("Panic! do_eigen_result was %i\n", do_eigen_result);
        return (FALSE);
    }
}
else
{
    do_the_discriminant_thing (permute_ctr, eigenval_ptr, eigenvec_ptr, 
        w_inv_m, *prior, current_var, number_of_vars, 
        dimension, current_cat_total, do_the_omission, increase);
}

return (TRUE);

} /* end "get_a_solution" */

/*====================== prepare_eigen_matrices =========================*/
int prepare_eigen_matrices (
                 MATRIX **original_eigenvalues, MATRIX **original_eigenvectors, 
                 MATRIX **original_w_inv_m_mat, MATRIX **best_w_inv_m_mat,
                 MATRIX **eigenvalues_real,     MATRIX **eigenvectors,
                 MATRIX **best_eigenvector,     
                 long classification, long first_time_through,
                 long number_of_classes, long number_of_cats)
{

if (first_time_through == FALSE)
{
    free (w->data); free (global_copy_of_w->data); 
    free (u->data); free (global_copy_of_u->data); 
    free (original_margin_holder->data);
    free (permuted_margin_holder->data);
    free ((*original_eigenvalues)->data); 
    free ((*original_eigenvectors)->data); 
    if (best_eigenvector != (MATRIX **) NULL)
        free ((*best_eigenvector)->data); 
    free ((*eigenvalues_real)->data);
    free ((*eigenvectors)->data);
    if (classification == DISCRIMINATION)
    {
        free ((*original_w_inv_m_mat)->data);
        free ((*best_w_inv_m_mat)->data);
    }
}
w                      = make_matrix (number_of_cats, number_of_cats,
                                    "W", REGULAR, ZERO_THE_MATRIX);
global_copy_of_w       = make_matrix (number_of_cats, number_of_cats,
                                    "Global W", REGULAR, ZERO_THE_MATRIX);
u                      = make_matrix (number_of_cats, number_of_cats,
                                    "U", REGULAR, ZERO_THE_MATRIX);
global_copy_of_u       = make_matrix (number_of_cats, number_of_cats,
                                    "Global U", REGULAR, ZERO_THE_MATRIX);
original_margin_holder = make_matrix ((long) number_of_classes + 1, 
                         number_of_cats, "Margin holder", 
                         REGULAR, ZERO_THE_MATRIX);
permuted_margin_holder = make_matrix ((long) number_of_classes + 1, 
                         number_of_cats, "Permuted holder", 
                         REGULAR, ZERO_THE_MATRIX);
if (classification == CLASSIFICATION)
{
    *original_eigenvectors = make_matrix (number_of_cats, number_of_cats,
                                    "Orig. E-vecs", REGULAR, ZERO_THE_MATRIX);
    *original_eigenvalues  = make_matrix (number_of_cats, 1L, "Orig. Vals",
                                    REGULAR, ZERO_THE_MATRIX);
    if (best_eigenvector != (MATRIX **) NULL)
         *best_eigenvector = make_matrix (1L, number_of_cats, "Best EigenVec",
                                      REGULAR, ZERO_THE_MATRIX);
    *eigenvalues_real      = make_matrix (number_of_cats, 1L, "EigenReal",
                                    REGULAR, ZERO_THE_MATRIX);
    *eigenvectors          = make_matrix (number_of_cats, number_of_cats,
                                    "Eigenvectors", REGULAR, ZERO_THE_MATRIX);
}
else
{
    *original_eigenvectors = make_matrix (number_of_classes, number_of_classes,
                                    "Orig. E-vecs", REGULAR, ZERO_THE_MATRIX);
    *original_eigenvalues  = make_matrix (number_of_classes, 1L, "Orig. Vals",
                                    REGULAR, ZERO_THE_MATRIX);
    if (best_eigenvector != (MATRIX **) NULL)
         *best_eigenvector = make_matrix (1L, number_of_classes, "Best EigVec",
                                      REGULAR, ZERO_THE_MATRIX);
    *eigenvalues_real     = make_matrix (number_of_classes, 1L, "EigenReal",
                                    REGULAR, ZERO_THE_MATRIX);
    *eigenvectors         = make_matrix (number_of_classes, number_of_classes,
                                    "Eigenvectors", REGULAR, ZERO_THE_MATRIX);
    *original_w_inv_m_mat = make_matrix (number_of_classes, number_of_classes,
                                    "W inv M", REGULAR, ZERO_THE_MATRIX);
    *best_w_inv_m_mat = make_matrix (number_of_classes, number_of_classes,
                                    "Best W inv M", REGULAR, ZERO_THE_MATRIX);
}

return (TRUE);
} /* end "prepare_eigen_matrices" */

/*====================== count_them_cats =========================*/
void count_them_cats (long number_of_vars, 
                     long *current_cat_total, long *currently_ordered, 
                     long *currently_numeric,
                     long *orderable_cats, MATRIX *c, Slong *increase)
{
long var_ctr;

*current_cat_total = 0L;
*currently_ordered = 0L;
*currently_numeric = 0L;
*orderable_cats    = 0L;

for (var_ctr = 0; var_ctr < number_of_vars; var_ctr++)
{
    if (am_i_in[var_ctr] >= CURRENTLY_IN) {
        *SUB (c, 0L, var_ctr) = 1.0;
        *current_cat_total += cats_in_var[var_ctr];
        if (increase[var_ctr] != UNORDERED
        &&  increase[var_ctr] != NUMERIC)
        {
            (*currently_ordered)++;
            *orderable_cats += cats_in_var[var_ctr];
        }
        if (increase[var_ctr] == NUMERIC)
            (*currently_numeric)++;
    }
    else {
            *SUB (c, 0L, var_ctr) = 0.0;
    }
}

} /* end "count_them_cats" */

/*========================= do_the_discriminant_thing ===================*/
int do_the_discriminant_thing (long permute_ctr,
       MATRIX *eigenval_ptr, MATRIX *eigenvec_ptr, MATRIX *original_w_inv_m_mat,
       MATRIX *prior, long i, 
       long number_of_vars, int dimension, 
       long current_cat_total, int do_the_omission, Slong *increase)
{
static MATRIX *temp_w, *reduced_w, *w_ptr;
static MATRIX *w_inv_m_mat = null_mat,
              *m_mat = null_mat, *h_mat = null_mat;
MATRIX *margin_ptr;
long length_of_work_1;
double *work_1;

/* Stuff for omit_count */
static long *omit_columns = (long *) NULL;
long omit_from_ctr, omit_to_ctr;
long omit_len = 0;

long var_ctr;
int inv_result;
#ifdef NAG
int NAG_status;
#else
long LAPACK_status;
char the_letter_V = 'V', the_letter_U = 'U';
#endif

if (current_cat_total > number_of_classes)
    length_of_work_1 = 3 * current_cat_total;
else
    length_of_work_1 = 3 * number_of_classes;
alloc_some_doubles (&work_1, length_of_work_1);


margin_ptr = original_margin_holder;
temp_w = make_matrix (w->nrow, w->ncol, "Temp w", REGULAR, TRUE);

matrix_copy (temp_w, w);
/*
** If there's more than one variable, then on the first permutation we
** set up the "reduced" stuff. Here we set up versions of the w, m,
** and w-inv-m matrices that omit the first column of each variable
** from two upward. (This makes w be of full rank.) The things named
** "ptr" point to the original matrices when dimension = 1, and to the
** reduced ones when dimension > 1.
*/
if (dimension > 1 && do_the_omission)
{
    if (permute_ctr == 0)
    {
        if (omit_columns != (long *) NULL)
            free (omit_columns);
/* The new set of columns has size "current_cat_total - dimension + 1" */
        omit_len = current_cat_total - dimension + 1;
        alloc_some_longs (&omit_columns, omit_len);
/* Set var_ctr to be the second variable that's at least CURRENTLY_IN. */
        var_ctr = 0;
        while (am_i_in[var_ctr] < CURRENTLY_IN)
            var_ctr++;
        var_ctr++;
        while (am_i_in[var_ctr] < CURRENTLY_IN)
            var_ctr++;
/*
** Now let's figure out which columns to omit. If var_ctr is already at
** the end of the number of variables, forget it. Otherwise, start
** "omit_from_ctr" counting from zero. Each column indexed by "omit_from_ctr"
** is included -- that is, gets put into "omit_columns" -- until we get 
** to the first category of a subsequent variable.
*/
        if (var_ctr < number_of_vars)
        {
            omit_to_ctr = 0;
            for (omit_from_ctr = 0; omit_from_ctr < omit_len; omit_from_ctr++)
            {
/* Put the column into "omit_columns," and increment "omit_to_ctr." */
                omit_columns[omit_from_ctr] = omit_to_ctr;
                omit_to_ctr ++;
/*
** Now if omit_to_ctr is one of the entries in cum_cats_this_subset,
** that means that column is the first category of its variable. So
** we move up omit_to_ctr without putting its number into omit_columns.
*/
                if (var_ctr < number_of_vars
                && omit_to_ctr == cum_cats_this_subset[var_ctr])
                {
                    omit_to_ctr ++;
/* Increment var_ctr (skipping those that aren't in); quit if necessary. */
                    var_ctr ++;
                    if (var_ctr >= number_of_vars)
                        continue;
                    while (am_i_in[var_ctr] < CURRENTLY_IN)
                    {
                        var_ctr++;
                        if (var_ctr >= number_of_vars)
                            continue;
                    }
                }
            }
        }
/* Make (or resize) these matrices to be the proper size. */
        resize_matrix (&reduced_w, omit_len, omit_len);
        w_ptr = reduced_w;
        resize_matrix (&w_inv_m_mat, omit_len, number_of_classes);
        resize_matrix (&original_w_inv_m_mat, omit_len, number_of_classes);
        resize_matrix (&m_mat, number_of_classes, omit_len);
    } /* end "if permute_ctr == 0" */
/* Copy just the non-omitted portion of w into w_ptr. */
    matrix_copy_portion (w_ptr, w, w_ptr->nrow, omit_columns, 
                         w_ptr->ncol, omit_columns);
    if (ridge > 0.0)
        matrix_ridge (w_ptr, ridge);
    inv_result = matrix_invert (w_ptr, (MATRIX *) NULL, 0);
    matrix_copy_portion (m_mat, margin_ptr, number_of_classes, (long *) NULL, 
                         omit_len, omit_columns);
} /* end "if dimension > 1" */
else
{
    if (permute_ctr == 0)
    {
        resize_matrix (&w_inv_m_mat, current_cat_total, number_of_classes);
        resize_matrix (&original_w_inv_m_mat, current_cat_total, 
                       number_of_classes);
        resize_matrix (&m_mat, number_of_classes, current_cat_total);
    }
    matrix_copy_portion (m_mat, margin_ptr, number_of_classes, (long *) NULL, 
                         current_cat_total, (long *) NULL);
    if (ridge > 0.0)
        matrix_ridge (temp_w, ridge);
/*
** Sometimes i = -1, as when we're getting a solution "one more time"
** (after we've established which is the best variable, and we "go
** back" to get the solution for that variable). We don't know the
** number of that variable, so we have to do the long inversion.
*/
    if (i < 0 || increase[i] == NUMERIC)
    {
        inv_result = matrix_invert (temp_w, (MATRIX *) NULL, 0);
    }
    else
    {
        inv_result = invert_diagonal (temp_w);
    }
    w_ptr = temp_w;
}

if (inv_result < 0)
{
    Rprintf ("Invert failed with %d\n", inv_result);
    return (-1);
}

matrix_multiply (w_ptr, m_mat, w_inv_m_mat, TRANSPOSE_SECOND);
if (permute_ctr == 0)
    matrix_copy (original_w_inv_m_mat, w_inv_m_mat);
resize_matrix (&h_mat, m_mat->nrow, w_inv_m_mat->ncol);
matrix_multiply (m_mat, w_inv_m_mat, h_mat, NO_TRANSPOSES);
divide_by_root_before_and_after (h_mat, prior);
scalar_multiply (h_mat, (MATRIX *) NULL, 
                 (double) (1.0 / (double) train_n_effective));

#ifdef NAG
NAG_status = 0;
f02abf_(h_mat->data, &(h_mat->nrow), &(h_mat->nrow), eigenval_ptr->data,
        eigenvec_ptr->data, &(h_mat->nrow), work_1, &NAG_status);
#else
/**
dsyev_ (&the_letter_V, &the_letter_U, &(h_mat)->nrow, h_mat->data, 
        &(h_mat->nrow),
        eigenval_ptr->data, work_1, &(length_of_work_1), &LAPACK_status);
**/
Rprintf ("About to call dsyev! This is exciting!\n");
F77_CALL(dsyev) (&the_letter_V, &the_letter_U, 
        (long *) &(h_mat)->nrow, h_mat->data, 
        (long *) &(h_mat->nrow),
        eigenval_ptr->data, work_1, &(length_of_work_1), &LAPACK_status);

free (work_1);

Rprintf ("We're back from dsyev! Was that so hard?\n");

matrix_copy (eigenvec_ptr, h_mat);
#endif

divide_by_root_prior (eigenvec_ptr, prior);

free (temp_w->data);

return (TRUE);
} /* end "do_the_discriminant_thing" */

/*========================= get_sequence_of_solutions ===================*/
int get_sequence_of_solutions (long quit_dimension,
    long number_of_vars, long permute, long permute_len, long *permute_indices,
    double improvement, long k_len, Slong *k,
    Slong *cats_in_var, Slong *cum_cats_ptr, MATRIX *c,
    MATRIX **original_eigenvectors, MATRIX **original_w_inv_m_mat,
    long *number_in_class, 
    long xval_ctr,
    MATRIX *xval_result, long *xval_ceiling,
    MATRIX *cost, MATRIX *prior, Slong prior_ind, Slong number_of_classes,
    double *misclass_rate, int do_the_omission, Slong *increase,
    long *once_out_always_out)
{
double biggest_permuted_eigenval, last_eigenvalue;
double relevant_eigenvalue; long relevant_eigenvalue_num;
double original_eigenvalue = 0.0;
double *current_eigenvalue;
long i, dimension, best_variable;
long how_many_bigger;
long *local_am_i_in;
int first_time_through = TRUE;
int quit;
int cat_ctr, var_ctr, k_ctr, permute_ctr;
int inv_result, do_nn_result;
int get_soln_result;
MATRIX *margin_ptr;
Slong *return_classes = (Slong *) NULL;

/* MATRIX *original_w_inv_m_mat; */

biggest_permuted_eigenval = -1.0;
last_eigenvalue = 0.0;
i = (long) -1;
dimension = 1L;
best_variable = -1L;
first_time_through = TRUE;
am_i_in_ctr = ALWAYS_IN;


alloc_some_doubles(&current_eigenvalue, number_of_vars);
alloc_some_longs(&local_am_i_in, number_of_vars);

for (var_ctr = 0; var_ctr < number_of_vars; var_ctr++)
{
/* Set up the (var_ctr)th entry of "am_i_in" and of "current_eigenvalue" */
    am_i_in[var_ctr] = CURRENTLY_OUT;
    current_eigenvalue[var_ctr] = -1.0;
}

quit = 0;

/***
misclass_mat = make_matrix (number_of_classes, number_of_classes, 
                     "Misclass mat", REGULAR, FALSE);
misclass_mat->data = misclass_data;
***/

while (!quit)
{
/* Move i up to the next variable that's not ALWAYS_IN, if there be one. */
    do {
        i++;
    } while (i < number_of_vars && am_i_in[i] >= ALWAYS_IN);
/* If i is past p, set it back to 1, increment dimension, and go again. */
    if (i >= number_of_vars)
    {
/*
** Find the largest among the eigenvalues we just looked at; then check
** to see if that variable should be made "always in."
*/
        relevant_eigenvalue     = -1.0;
        relevant_eigenvalue_num = -1L;
        i = -1.0;
        for (cat_ctr = 0L; cat_ctr < number_of_vars; cat_ctr++)
        {
            if (am_i_in[cat_ctr] == ALWAYS_OUT 
            ||  am_i_in[cat_ctr] >= ALWAYS_IN)
                continue;
            if (current_eigenvalue[cat_ctr] > relevant_eigenvalue) {
                relevant_eigenvalue     = current_eigenvalue[cat_ctr];
                relevant_eigenvalue_num = cat_ctr;
            }
        }

/* 
** What if the number of rows is equal to the dimension? (This will
** happen in classification if every variable in the model is numeric, 
** and there's only one knot per variable.) We want to keep going, I 
** guess, so let's set "last_eigenvalue" to something small.
*/
        if (classification == CLASSIFICATION 
        &&  dimension == eigenvec_ptr->nrow)
            last_eigenvalue = -2.0;
/* Here we're using TSS as eigenvalues. It's positive. */
/******* OLD WAY: WITH IMPROVEMENT ***********/
/***
Rprintf ("Check rel num %i, permute %i, improve %f, rel val %f vs. %f\n",
relevant_eigenvalue_num, permute, improvement, relevant_eigenvalue,
biggest_permuted_eigenval);
***/
        if (relevant_eigenvalue_num < 0
        || (permute == 0 
            && relevant_eigenvalue - last_eigenvalue < improvement)
        || (permute != 0 
            && relevant_eigenvalue < biggest_permuted_eigenval))
        {
/*
** There's no improvement. We want to drop back down to the last 
** successful set of variables. "Am_i_in" is already correct.
** "Xval_ceiling" gives the largest dimension that we have considered
** at every xval. These are the only ones we have good error estimates
** for. If xval_ceiling_ind is >= 1, we use that number, even if it means
** we need to include variables that don't't help on this xval. If xval_
** ceiling_ind is = 0, it means use as the ceiling the number of variables
** that entered on the first xval. If it's -1, use the smallest number
** that entered on any xval.
*/
            if (verbose > 1)
                Rprintf ("That's it; we're done, no new additions!\n");
            dimension --;
            if (xval_ceiling_ind < 1) {
                if (xval_ctr == 0 
                || (xval_ceiling_ind == -1 && dimension < *xval_ceiling))
                    *xval_ceiling = dimension; 
                break;
            }
            else
                if (verbose > 1)
                    Rprintf ("No stop: dim %i, ceiling %i, ind %i\n",
                        dimension, xval_ceiling, xval_ceiling_ind);
        }

        if (verbose > 1)
            Rprintf (
                "\t\t*** Okay; %ld is always in: improvement %f!\n",
            relevant_eigenvalue_num, relevant_eigenvalue - last_eigenvalue);
        am_i_in[relevant_eigenvalue_num] = am_i_in_ctr++;
        last_eigenvalue = relevant_eigenvalue;
/*****
if (am_i_in_ctr > 1) {
Rprintf ("Artificial break!\n");
print_matrix (global_copy_of_w, (0x10) | (0x40));
print_matrix (global_copy_of_u, (0x10) | (0x40));
print_matrix (*original_eigenvectors, (0x10) | (0x40));
print_matrix (eigenvalues_real, 8);
return (FALSE);
}
*****/
/*
** Here's where we get a cross-validated estimate of the misclassification
** error. For each cross-validation piece (the number of which has been
** specified by "xvals"), we call "do_nn" with that piece as the test set
** and everything else as the training set. From that call we get a
** vector of misclassification rates, one for each entry in "k". That
** vector gets added to coresponding row of this entry of xval_result.
** When this entry of xval_result is finished, we go on. When no more
** variables are to be added, we go back and find the right combination
** of variables and nearest-neighbors, and use that on the real test set.
**
** "Best_eigenvector" holds the best just from the most recently-added
** variable. ("Best" means "best among the permutations.") So we need
** to get the solution associated with this set of variables. If
** the most recent one looked at happened to be best, we needn't do
** this, but there's no easy way to tell right now.
** 
*/
        count_them_cats (number_of_vars, &current_cat_total, &currently_ordered,
                         &currently_numeric, &orderable_cats, c, increase);

        prepare_eigen_matrices (&original_eigenvalues, original_eigenvectors, 
                                original_w_inv_m_mat,  &best_w_inv_m_mat,
                                &eigenvalues_real,     &eigenvectors,
                                &best_eigenvector,     
                                classification, first_time_through, 
                                number_of_classes, current_cat_total);

        margin_ptr = original_margin_holder;

        get_soln_result = get_a_solution (0L, permute_indices, train_n,
                        (long) -1, margin_ptr, training, cats_in_var, 
                        cum_cats_ptr, knots,
                        number_in_class,
                        original_eigenvalues, *original_eigenvectors, 
                        *original_w_inv_m_mat, ridge, 
                        classification, number_of_vars, dimension, 
                        current_cat_total, &prior, prior_ind, number_of_classes,
                        FALSE, do_the_omission, increase, FALSE);

        if (get_soln_result == FALSE)
        {
            Rprintf ("Panic! get_soln_result was FALSE\n");
            return (FALSE);
        }

#if 0
if (first_time_through)
{
    print_matrix (global_copy_of_w, 8);
    print_matrix (global_copy_of_u, 8);
    print_matrix (*original_w_inv_m_mat, 8);
    print_matrix (original_eigenvalues, 8);
}
#endif
        
        if (dimension <= currently_numeric)
            matrix_extract (*original_eigenvectors, ROW,
                (*original_eigenvectors)->nrow - 1, best_eigenvector, FALSE);
        else
            matrix_extract (*original_eigenvectors, ROW,
                (*original_eigenvectors)->nrow - 2, best_eigenvector, FALSE);

        if (first_time_through)
            print_matrix (best_eigenvector, 8);

        if (currently_ordered > 0)
            add_ordered_variable (&relevant_eigenvalue,
                relevant_eigenvalue_num,
                dimension, cats_in_var, best_eigenvector, increase);

        if (quit_dimension == dimension)
        {
            if (verbose > 1)
                Rprintf ("Quitting because quit dim = dim = %i\n",dimension);
            break;
        }

        if (xval_result != (MATRIX *) NULL)
        {
            if (classification == CLASSIFICATION)
            {
/***
                misclass_mat = make_matrix (number_of_classes, 
                               number_of_classes, 
                               "Misclass mat", REGULAR, ZERO_THE_MATRIX);
***/
                return_classes = &Slong_FALSE;
                do_nn_result = do_nn (&Slong_FALSE, training, training,
                   c, k, &k_len, &Slong_TRUE, /* theyre_the_same */
                   SUB (best_eigenvector, 0L, 0L),
                   cats_in_var, cum_cats_ptr, knots, cost,
                   &prior_ind, prior, misclass_rate, NULL, return_classes, 
                   (Slong *) NULL,
                   &xval_lower, &xval_upper, xval_indices,
                   increase, &number_of_classes, &verbose);
               if (do_nn_result == FALSE)
               {
                   Rprintf ("Panic! do_nn_result was FALSE!\n");
                   return (FALSE);
               }

            }
            else
            {
/* Don't worry about "best_eigenvector"; no ordering in discrimination.*/

/***
                misclass_mat = make_matrix (number_of_classes, 
                               number_of_classes, 
                               "Misclass mat", REGULAR, ZERO_THE_MATRIX);
***/

                do_discriminant (training,
                    original_eigenvalues, *original_eigenvectors,
                    *original_w_inv_m_mat, cats_in_var, cum_cats_ptr,
                    dimension, number_of_vars, knots, cost,
                    prior, misclass_rate, misclass_mat, do_the_omission);

/**
    Rprintf ("Discrimination misclass_rate is %f\n", *misclass_rate);
                print_matrix (misclass_mat, 8);
                free (misclass_mat->data);
**/
            }

            for (k_ctr = 0; k_ctr < k_len; k_ctr++)
            {
/*
Rprintf ("Adding %f to spot %ld %ld\n", misclass_rate[k_ctr],
 dimension - 1, k_ctr);
*/
                *SUB (xval_result, dimension - 1, k_ctr) =
                    *SUB (xval_result, dimension - 1, k_ctr) 
                    + misclass_rate[k_ctr];
            }
/*
Rprintf ("Xval result is now...\n");
print_matrix (xval_result, 8);
*/
        } /* end "if xval_result is not NULL" */
/*
** Set i to be the next variable that's currently out.
*/
        for (cat_ctr = 0L; cat_ctr < number_of_vars; cat_ctr++)
        {
            if (am_i_in[cat_ctr] == CURRENTLY_OUT && i < 0)
            {
                i = cat_ctr;
                break;
            }
        }

        biggest_permuted_eigenval = -1.0;
        dimension ++;
    } /* end "if i >= number_of_vars" */
/*
** If there's no new i, we're done with this iteration. Reset ceiling. 
*/
    if (i < 0)
    {
        dimension --;
        if (xval_ceiling_ind < 1) {
            if (xval_ctr == 0 
            || (xval_ceiling_ind == -1 && dimension < *xval_ceiling))
                *xval_ceiling = dimension; 
            if (verbose > 1)
                Rprintf ("No new i; let's break, baby\n");
            break;
        }
        else
            if (verbose > 1)
                Rprintf ("No stop 2: dim %i, ceiling %i, ind %i\n",
                    dimension, xval_ceiling, xval_ceiling_ind);
    }
    if (xval_ctr != 0 && dimension > *xval_ceiling)
    {
        if (verbose > 1)
            Rprintf ("Reached ceiling on loop %ld\n", xval_ctr);
        break;
    }

    am_i_in[i] = CURRENTLY_IN;

    count_them_cats (number_of_vars, &current_cat_total, &currently_ordered, 
                     &currently_numeric, &orderable_cats, c, increase);

    how_many_bigger = 0;
    inv_result = 0;
    delete_this_variable_forever = FALSE;

/*======================= Permute loop ============================*/
    for (permute_ctr = 0; permute_ctr <= permute; permute_ctr++)
    {
        if (inv_result < 0)
        {
            Rprintf ("Giving up on variable %li\n", i);
            how_many_bigger = permute_tail + 1;
            break;
        }

        if (first_time_through == TRUE)
        {
            deal_with_missing_values (training, missing_max, increase, 
                                      cats_in_var, missing_values);
            insert_missing_values (training, missing_max, missing_values);
        }

        if (permute_ctr == 0)
        {
            prepare_eigen_matrices (
                 &original_eigenvalues, original_eigenvectors, 
                 original_w_inv_m_mat,  &best_w_inv_m_mat,
                 &eigenvalues_real,     &eigenvectors,
                 &best_eigenvector, 
                 classification, first_time_through, 
                 number_of_classes, current_cat_total);
            margin_ptr   = original_margin_holder;
            eigenvec_ptr = *original_eigenvectors;
            eigenval_ptr = original_eigenvalues;
        }
        else
        {
            margin_ptr   = permuted_margin_holder;
            eigenvec_ptr = eigenvectors;
            eigenval_ptr = eigenvalues_real;
        }

        get_soln_result = get_a_solution (permute_ctr, permute_indices, 
                        permute_len, i,
                        margin_ptr, training, cats_in_var, cum_cats_ptr, knots,
                        number_in_class,
                        eigenval_ptr, eigenvec_ptr, 
                        *original_w_inv_m_mat, ridge, 
                        classification, number_of_vars, dimension, 
                        current_cat_total, 
                        &prior, prior_ind, number_of_classes,
                        first_time_through, do_the_omission, increase, FALSE);
        if (get_soln_result == FALSE)
        {
            Rprintf ("Panic! get_soln_result was FALSE\n");
            return (FALSE);
        }

        first_time_through = FALSE;

/*
** Grab the second eigenvalue from the end.  In the discrimination case
** the "largest eigenvalue" is actually the sum of eigenvalues from
** there to the end. We add them all up and subtract the one with the
** highest number, which is "nrow - 1".
** Weird exception: if there's only one row (which can happen if
** this is a numeric variable with only one knot), use that one.
*/
        if (dimension <= currently_numeric && cats_in_var[i] == 1)
        {
            largest_eigenvalue_num = eigenval_ptr->nrow - 1;
            largest_eigenvalue     = *SUB (eigenval_ptr, 0, 
                                           largest_eigenvalue_num);
        }
        else
        {
            largest_eigenvalue_num = eigenval_ptr->nrow - 2;
/***
            if (eigenval_ptr->data[largest_eigenvalue_num + 1] > 1.0)
            {
                Rprintf ("Panic! Eigenvalue %i is %f!\n",
                    largest_eigenvalue_num + 1, 
                    eigenval_ptr->data[largest_eigenvalue_num + 1]);
                return (FALSE);
            }
***/
            if (classification == CLASSIFICATION) 
                largest_eigenvalue     = *SUB (eigenval_ptr, 0, 
                                               largest_eigenvalue_num);
             else
                largest_eigenvalue     = matrix_sum (eigenval_ptr) -
                                     eigenval_ptr->data[eigenval_ptr->nrow -1];
        }

        if (permute_ctr == 0)
        {
            original_eigenvalue   = largest_eigenvalue;
            original_relevant_num = largest_eigenvalue_num;
            if (verbose > 3)
            {
                Rprintf ("i: %ld Original (%ld) ", 
                                 i, largest_eigenvalue_num);
                Rprintf (" is %f\n", largest_eigenvalue);
            }
            if (permute == 0 && original_eigenvalue < improvement 
            &&  *once_out_always_out == TRUE)
            {
                if (verbose > 3)
                    Rprintf ("%li is always out!\n", i);
                inv_result = -1;
                delete_this_variable_forever = TRUE;
                continue;
            }
        }
        else
        {
            if (verbose > 4)
            {
                Rprintf ("Permuted (%ld) ", largest_eigenvalue_num);
                Rprintf (" is %f\n", largest_eigenvalue);
            }
        }
                
        if (largest_eigenvalue > original_eigenvalue)
        {
        /*    Rprintf ("Permutation %ld is bigger!\n", permute_ctr); */
            if (largest_eigenvalue > biggest_permuted_eigenval)
                biggest_permuted_eigenval = largest_eigenvalue;
             how_many_bigger++;
        }
/*
        else
            Rprintf ("Perm. %ld; orig is larger than largest!\n");
*/

    } /* end "permute_ctr" loop */

/***
    Rprintf ("tColumn %li had %li bigger\n", i, how_many_bigger);

    if (classification == CLASSIFICATION && ridge <= 0)
    {
        free (eigenvalues_imaginary->data); free (eigenvalues_imaginary);
        free (eigenvalues_beta->data); free (eigenvalues_beta);
    }

    if (classification == DISCRIMINATION)
    {
        free (h_mat->data);
    }
***/

    if (delete_this_variable_forever)
    {
        am_i_in[i] = ALWAYS_OUT;
        delete_this_variable_forever = FALSE;
        if (verbose > 1)
            Rprintf ("Deleting variable %li forever!\n", i);
    }
    else
        am_i_in[i] = CURRENTLY_OUT;

/*
Rprintf ("Increase %i; it's %i; %i bigger with tail %i\n", 
i, increase[i], how_many_bigger, permute_tail);
*/
    
    if (increase[i] == UNORDERED
    ||  increase[i] == NUMERIC)
    {
        if (how_many_bigger >= permute_tail)
        {
            if (verbose > 1)      
                Rprintf ( "Variable %ld fails the permutation test!\n", i);
            if (*once_out_always_out)
            {
                am_i_in[i] = ALWAYS_OUT;
                if (verbose > 1)
                    Rprintf ("Deleting variable %li forever!\n", i);
            }
            current_eigenvalue[i] = -1.0;
        }
        else
        {
            current_eigenvalue[i] = original_eigenvalue;
            if (best_variable < 0 || best_eigenvalue < original_eigenvalue)
            {
                best_variable     = i;
                best_eigenvalue   = largest_eigenvalue;
                best_relevant_num = largest_eigenvalue_num;
                if (classification == CLASSIFICATION)
                    matrix_extract (*original_eigenvectors, ROW,
                        largest_eigenvalue_num, best_eigenvector, FALSE);
                else
                {
                    resize_matrix (&best_w_inv_m_mat, 
                        (*original_w_inv_m_mat)->nrow, 
                        (*original_w_inv_m_mat)->ncol);
                    matrix_copy (best_w_inv_m_mat, *original_w_inv_m_mat);
                }
            }
        }
        continue;
    }

/* 
** If we get here, it's an ordered variable. First see if it passes
** the permutation test.
*/

    if (how_many_bigger >= permute_tail)
    {
        current_eigenvalue[i] = -1.0;
        if (*once_out_always_out)
            am_i_in[i] = ALWAYS_OUT;
    }
    else
    {
/**     current_eigenvalue[i] = objective_result; **/
        current_eigenvalue[i] = original_eigenvalue;
        if (best_variable < 0 || best_eigenvalue < original_eigenvalue)
        {
            best_variable     = i;
            best_eigenvalue   = largest_eigenvalue;
            best_relevant_num = largest_eigenvalue_num;
            if (classification == CLASSIFICATION)
                matrix_extract (*original_eigenvectors, ROW,
                    largest_eigenvalue_num, best_eigenvector, FALSE);
            else
            {
                resize_matrix (&best_w_inv_m_mat, 
                    (*original_w_inv_m_mat)->nrow, 
                    (*original_w_inv_m_mat)->ncol);
                matrix_copy (best_w_inv_m_mat, *original_w_inv_m_mat);
            } /* end "else," i.e. if discrimination */
        } /* end "if this is the first or best variable" */
    } /* end "else," i.e. if permute_test didn't fail */
} /* end "while !quit" */


free (current_eigenvalue);
/* free (am_i_in); */

get_a_solution ((long) 0, (long *) NULL, (long) 0, (long) 0, 
    (MATRIX *) NULL, (MATRIX *) NULL, (Slong *) NULL, (Slong *) NULL, 
    (double **) NULL, (long *) NULL, (MATRIX *) NULL, (MATRIX *) NULL, 
    (MATRIX *) NULL, (double) 0, classification, (long) 0, (int) 0, 
    (long)  0, (MATRIX **) NULL, (Slong)  0, (int)  0, (int)  0, 
    (int) 0, (Slong *) NULL, TRUE /* quit */);

return (TRUE);

} /* end "get_sequence_of_solutions" */
