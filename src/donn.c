/*
**      do_nn ()
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

long verbose;
long save_verbose;
long global_test_ctr;
long number_of_classes;
long *xval_indices;
long xval_lower;
long xval_upper;
Slong *increase;

double c_euclidean (double *, double *, double *, long, double);
double f_euclidean (double *vec_1, double *vec_2, double *phi, double *c, 
                    long n, double threshold, Slong *cats_in_var, 
                    Slong *cum_cats,
                    MATRIX *prior, double **knots);
double c_absolute (double *, double *, double *, long, double);

/* Statics for "xpoll" */
static double *class_results;
static double *class_results_with_ties;
static int *tie_marker;
void xpoll (long *classes, double *distances, Slong *k, long how_many_ks,
          long largest_k, long slots, MATRIX * prior, long *outcome);
/*=========================== do_nn =================================*/

int do_nn (Slong *quit, MATRIX *training, MATRIX *test, 
           MATRIX *c, Slong *k, long *in_how_many_ks, 
           Slong *theyre_the_same, double *phi, Slong *cats_in_var, 
           Slong *cum_cats_this_subset,
           double **knots, MATRIX *cost, 
           Slong *prior_ind, MATRIX *prior, double *error_rates,
           MATRIX *misclass_mat, Slong *return_classes, Slong *classes,
           long *in_xval_lower, long *in_xval_upper, long *in_xval_indices,
           Slong *in_increase, Slong *in_number_of_classes, Slong *in_verbose)
{
/*
** This deluxe version of do_nn became necessary when we started to
** do cross-validation. It used to be that we would "adjust" the
** matrices before we got here, by replacing their entries with
** the "phi" values we computed. But we can't do that any more,
** because we need to preserve the matrices for later use. So we
** pass the "phi" values in, plus the instructions on how to use
** them (cats_in_var, cum_cats_this_subset, increase, and knots).
**
** Then we use this info inside "f_euclidean" ("f" for "fancy")
** to compute nearest neighbors.
**
** Actually, the deluxe-ness goes even farther than that. If the thing
** called "xval_indices" isn't NULL, then for this function we consider
** the test set to be only those entries in the test set whose row
** numbers are greater than or equal to "xval_lower" and less than 
** "xval_upper." The training set is likewise things whose numbers
** are smaller than "xval_lower" or greater than or equal to "xval_upper."
** 
** This really only makes sense if the training and test sets are the
** same file.
*/
long j, train_ctr, test_ctr;
long test_item_count;
long dist_ctr, move_ctr, k_ctr,
       number_of_nearest,
       test_class;
long how_many_ks, number_of_vars;

double dist;

static int initialized = 0;
static double *nearest_distance;
static long *nearest_class;
static long *nearest_neighbor;
static long *poll_result;
static long *misclass_with_distance_zero;
static long largest_k;
static long slots;

if (*quit == TRUE)
    initialized = TRUE;

if (*prior_ind == IGNORED)
    prior = (MATRIX *) NULL;

how_many_ks       = *in_how_many_ks;

if (initialized == FALSE)
{
    largest_k = -1L;
    for (j = 0; j < how_many_ks; j++)
        if (k[j] > largest_k)
            largest_k = k[j];
    
    slots = largest_k + 15;

    if (alloc_some_doubles (&nearest_distance, (unsigned long) slots) != 0)
    {
        Rprintf ("Unable to get %li doubles 4 dists; abort\n", slots);
        return (0);
    }
    
    if (alloc_some_longs (&nearest_neighbor, (unsigned long) slots) != 0)
    {
        Rprintf ("Unable to get %li longs 4 neighbors; abort\n", slots);
        return (0);
    }
    
    if (alloc_some_longs (&nearest_class, (unsigned long) slots) != 0)
    {
        Rprintf ("Unable to get %li longs for classes; abort\n", slots);
        return (0);
    }

    if (alloc_some_longs (&poll_result, (unsigned long) how_many_ks) != 0)
    {
        Rprintf ("Unable to get %li longs for results; abort\n", 
                 how_many_ks);
        return (0);
    }

    if (alloc_some_longs (&misclass_with_distance_zero, 
              (unsigned long) how_many_ks) != 0)
    {
        Rprintf ("Unable to get %li longs for miss II; abort\n", 
                 how_many_ks);
        return (0);
    }

    for (j = 0; j < how_many_ks; j++)
    {
        error_rates[j] = 0.0;
        misclass_with_distance_zero[j] = 0L;
    }

    initialized = TRUE;
}

if (*quit == TRUE)
{
    free (nearest_distance);
    free (nearest_neighbor);
    free (nearest_class);
    free (poll_result);
    free (misclass_with_distance_zero);
    free (class_results);
    free (class_results_with_ties);
    free (tie_marker);
    initialized = FALSE;
    return (TRUE);
}

xval_lower        = *in_xval_lower;
xval_upper        = *in_xval_upper;
xval_indices      =  in_xval_indices;
increase          =  in_increase;
number_of_classes = *in_number_of_classes;
verbose           = *in_verbose;

number_of_vars = training->ncol - 1L;
/*
** Zero out the misclass_matrix, if there is one. 
*/
if (misclass_mat != (MATRIX *) NULL)
    zero_matrix (misclass_mat);

test_item_count = 0L;
/*
** Now we go through the test set...
*/
for (test_ctr = 0; test_ctr < test->nrow; test_ctr++)
{
    global_test_ctr = test_ctr;
    if (xval_indices != (long *) NULL)
    {
            if ((xval_lower < xval_upper)
            && (test_ctr < xval_lower || test_ctr >= xval_upper))
            {
                continue;
            } 
    }
    test_item_count++;
    if (verbose > 3 && test_ctr <= 1)
        Rprintf ("Computing dists for test record %li\n", test_ctr);
/*
** ..zero out the nearest neighbor information...
*/
    for (j = 0; j < slots; j++)
    {
        nearest_neighbor[j] = (long) -1;
        nearest_distance[j] = -1.0;
        nearest_class[j]    = (long) -1;
    }
    if (test_ctr <= 1 && verbose > 3) 
        Rprintf ("-1's into all %i slots (largest k is %i)\n",
            slots, largest_k);
    number_of_nearest = 0;
/*
** ... and compute the distance from this record to each of the training
** set records, in turn.
*/
    for (train_ctr = 0; train_ctr < training->nrow; train_ctr++)
    {
        if (xval_indices != (long *) NULL)
        {
                if (train_ctr >= xval_lower && train_ctr < xval_upper)
                    continue;
        }
        if (*theyre_the_same && test_ctr == train_ctr)
            continue;
            
if (verbose > 3 && test_ctr <= 1 && train_ctr <= 1)
    Rprintf ("Test/train %i %i, threshold %f\n", test_ctr, train_ctr,
                            nearest_distance[largest_k]);

        dist = f_euclidean (SUB (training, train_ctr, 1), 
                            SUB (test, test_ctr, 1), phi,
                            SUB (c, 0, 0),
                            number_of_vars, nearest_distance[largest_k],
                            cats_in_var, cum_cats_this_subset,
                            prior, knots);
/***
double f_euclidean (double *vec_1, double *vec_2, double *phi, double *c, 
                    long n, double threshold, long *cats_in_var, 
                    long *cum_cats_this_subset, long *increase, 
                    double **knots)

        if (dist_function == EUCLIDEAN)
            dist = c_euclidean (SUB (training, train_ctr, 1), 
                                SUB (test, test_ctr, 1),
                                SUB (c, 0, 0),   
                                number_of_vars, nearest_distance[largest_k]);
        else
            dist = c_absolute (SUB (training, train_ctr, 1), 
                               SUB (test, test_ctr, 1), 
                               SUB (c, 0, 0), number_of_vars, 
                               nearest_distance[largest_k]);
***/
/*
** Each distance function is given a threshold, which is the largest of the
** "nearest-neighbor" distances. As soon as a distance gets above that, we
** know we can stop this comparison. The function returns -1, and we continue.
*/
        if (verbose > 3 && test_ctr <= 1 && train_ctr <= 1 && dist < 0)
            Rprintf ("dist was < 0, skip\n");
        if (dist < 0)
            continue;
/*
** If this distance is smaller than the largest on our list, or if we haven't
** encountered "slots" records yet, begin the nearest neighbor processing.
*/
        if (dist < nearest_distance[slots-1] || number_of_nearest < slots)
        {
            if (verbose > 4 && dist <= nearest_distance[0])
                Rprintf ("Smallest so far for %li is %li, distance %f\n", 
                         test_ctr, train_ctr, dist);
/*
** Find the spot for this new neighbor. When we find it (and by "the spot"
** I mean the smallest current neighbor bigger than this distance) we move
** everybody from the spot forward one in the list and insert the new entry
** in that spot. Make sure to save the largest distance.
*/
            for (dist_ctr = 0; dist_ctr < slots; dist_ctr ++)
            {
/*
** If the current "nearest_distance" isn't a -1, and if it's smaller
** than "dist," move up to the next "nearest_distance. 
*/
                if (nearest_distance[dist_ctr] >= 0.0 &&
                    dist >= nearest_distance[dist_ctr])
                    continue;
/*
** Otherwise, move everything from spot (i-1) to spot i,  and put
** information for this record into the appropriate spot.
*/
                for (move_ctr = slots - 1; move_ctr > dist_ctr; move_ctr --)
                {
                    nearest_distance[move_ctr] = 
                        nearest_distance[move_ctr - 1];
                    nearest_class[move_ctr] = nearest_class[move_ctr - 1];
                    nearest_neighbor[move_ctr] = 
                        nearest_neighbor[move_ctr - 1];
                }
                nearest_distance[dist_ctr] = dist;
                nearest_neighbor[dist_ctr] = train_ctr;
                nearest_class[dist_ctr] = (long) *SUB (training, train_ctr, 0);
                break;
            } /* end "for" loop on nearest neighbor arrays. */
            if (number_of_nearest < slots)
                number_of_nearest ++;
        } /* end "if this is a nearest neighbor" */
    } /* end "for" for looping over the training set. */


        if (verbose >= 2 && test_ctr <= 1)
        {
            Rprintf ("Test rec %li (a %li) has closest ", test_ctr, 
                          (long) *SUB (training, train_ctr, 0));
            /* for (j = 0; j < slots; j++) */
            for (j = 0; j < 1; j++)
            {
                Rprintf ( "(%i:) rec. %li, class %li    ", 
                    j, nearest_neighbor[j], nearest_class[j]);
            }
            Rprintf ("\n");
        }


    if (test_ctr == 0)
        for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
        {
            poll_result[k_ctr] = 0L;
        }

    if (verbose > 3 && test_ctr <= 1)
    {
        for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
            Rprintf ("nearest_class[k_ctr] is %i...", nearest_class[k_ctr]);
    }

    if (verbose >= 2 && test_ctr <= 1)
    {
        Rprintf ("About to poll for test record %li\n", test_ctr);
        for (j = 0; j < 5; j++)
        {
            Rprintf ("%li: %li: nn %li, class %li, dist %lf\n", 
                test_ctr, j, nearest_neighbor[j], 
                nearest_class[j], nearest_distance[j]);
        }
    }
    save_verbose = verbose;
    if (verbose >= 2 && test_ctr <= 1) verbose = 4;
    if (verbose >= 4)
    {
        if (prior == (MATRIX *) NULL)
            Rprintf ("About to call xpoll, and prior is so very NULL\n");
        else
            Rprintf ("About to call xpoll, and prior is so very *not* NULL\n");
}
    xpoll (nearest_class, nearest_distance, k, how_many_ks, 
          largest_k, slots, prior, poll_result);
    verbose = save_verbose;

    if (verbose >= 2 && test_ctr <= 1)
    {
        for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
            Rprintf (", poll_result is %i\n", poll_result[k_ctr]);
    }
    
    test_class = (long) *SUB (test, test_ctr, 0);
/*
** If "return_classes" is true, take the first poll result, since 
** presumably there's only one specific k supplied anyway.
*/
    if (*return_classes == TRUE)
        classes[test_ctr] = (Slong) poll_result[0];

    for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
    {
        if (misclass_mat != (MATRIX *) NULL)
            *SUB (misclass_mat, test_class, poll_result[k_ctr])
            = *SUB (misclass_mat, test_class, poll_result[k_ctr]) + 1;
        if (poll_result[k_ctr] != test_class)
        {
/*
** If a cost matrix was passed in, use it. The relevant entry is the one
** with the true class as the row index and the prediction as the column.
*/
            if (cost == (MATRIX *) NULL)
                error_rates[k_ctr]++;
            else
                error_rates[k_ctr] 
                    += *SUB (cost, test_class, poll_result[k_ctr]);
            if (nearest_distance[0] == 0.0) 
                misclass_with_distance_zero[k_ctr]++;
            if (test_ctr <= 1 && verbose > 2 && *theyre_the_same == FALSE) {
                Rprintf (
"k = %ld: Classif.err test rec. %li (a %li) as %li (nearest: %li, dist. %f)", 
                k[k_ctr], test_ctr, test_class, (long) poll_result[k_ctr],
                (long) nearest_neighbor[0], nearest_distance[0]);
                Rprintf ("(next: %li, dist. %f), (next: %li, dist. %f)\n",
                (long) nearest_neighbor[1], nearest_distance[1],
                (long) nearest_neighbor[2], nearest_distance[2]);
                }
        }
        else
        {
            if (verbose > 2 && *theyre_the_same == FALSE)
                Rprintf (
"k = %ld: Classif.ok test rec. %li (a %li) as %li (nearest: %li, dist. %f)\n", 
                k[k_ctr], test_ctr, test_class, (long) poll_result[k_ctr],
                (long) nearest_neighbor[0], nearest_distance[0]);
        }
    }

} /* end "for i" loop for test set. */

for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
{
    error_rates[k_ctr] /= (double) test_item_count;
    if (verbose > 1)
    {
        Rprintf ("k = %ld: misfrac. %f records (of %li); %li (%f) dist 0\n",
            k[k_ctr], error_rates[k_ctr], test_item_count,
            misclass_with_distance_zero[k_ctr], 
                 ((double) misclass_with_distance_zero[k_ctr]) 
                                          / ( (double) test_item_count));
    }
}

/*
** The responsible user will have freed this memory with a call
** with initialized=-TRUE, so this is unnecessary.
**/ 
/***
free (nearest_distance);
free (nearest_neighbor);
free (nearest_class);
free (poll_result);
** free (misclass); **
free (misclass_with_distance_zero);
initialized = FALSE;
***/

return (TRUE);

} /* end "do_nn." */

/*============================  poll  =====================================*/
void xpoll (long *classes, double *distances, Slong *k, long how_many_ks,
          long largest_k, long slots, MATRIX * prior, long *outcome)
{
int i, k_ctr;
int tie;
int max_count;
int max_class = -1;

static int initialized = 0;

if (initialized == FALSE)
{
    if (alloc_some_doubles (&class_results, number_of_classes) != 0)
    {
        Rprintf ("Couldn't get %li doubles for poll results; abort\n", 
            number_of_classes);
    }
    if (alloc_some_doubles (&class_results_with_ties, number_of_classes) != 0)
    {
        Rprintf ("Couldn't get %li ints for poll results; abort\n", 
            number_of_classes);
    }
    if (alloc_some_ints (&tie_marker, number_of_classes) != 0)
    {
        Rprintf ("Couldn't get %li ints for ties in poll; abort\n", 
            number_of_classes);
    }
    initialized = TRUE;
}

/* Zero out the results and tie markers arrays. */
for (i = 0; i < number_of_classes; i++)
{
    class_results[i] = 0.0;
    class_results_with_ties[i] = 0.0;
    tie_marker[i] = 0;
}

/* Okay. Now we go through the list of k's. First of all, if any
** k is 1 or 2, return the class of the nearest neighbor. (This is
** clear for k = 1. For k = 2, ties are broken by the first nearest
** neighbor anyway.)
*/

for (k_ctr = 0; k_ctr < how_many_ks; k_ctr++)
{
/*** This is wrong. These things can be ties.
    if (k[k_ctr] == 1 || k[k_ctr] == 2)
    {
        outcome[k_ctr] = classes[0];
        continue;
    }
****/


/* Zero out the "class_results" array ... */
    for (i = 0; i < number_of_classes; i++)
    {
        class_results[i] = 0.0;
    }
if (verbose >= 4)
Rprintf ("Number of classes is %li\n", number_of_classes);
if (verbose >= 4)
Rprintf ("First two class results are %f and %f\n",
class_results[0], class_results[1]);
    
/* ...and go through the neighbors to fill it up again. When classes[i]
** = j, add one to the j-th entry of class_results. Well, not one,
** exactly; if priors isn't NULL, add 1/(that class' prior). That
** way, classes with large priors contribute less. Which is as it should be.
*/
if (verbose >= 4)
{
    if (prior == (MATRIX *) NULL)
        Rprintf ("By the way, prior is so very NULL\n");
    else
        Rprintf ("By the way, prior is so very *not* NULL\n");
}
    for (i = 0; i < k[k_ctr]; i++)
    {
        if (prior == (MATRIX *) NULL)
            class_results[classes[i]] ++;
        else
            class_results[classes[i]] +=
                 (1.0 / *SUB (prior, classes[i], classes[i]));
    }
if (verbose >= 4)
Rprintf ("First two class results are %f and %f\n",
class_results[0], class_results[1]);
/*
** Okay. It could happen that some other neighbors are tied
** with the kth one. We would know that if their distances equalled the
** kth distance. First copy the "class_results" array to the handy
** "class_results_with_ties"; then look through the remaining neighbors 
** (there are "slots" neighbors) to see if any of them should be counted.
*/

    for (i = 0; i < number_of_classes; i++)
        class_results_with_ties[i] = class_results[i];

    i = k[k_ctr];

    while (i < slots && distances[i] == distances[k[k_ctr]-1])
    {
        if (prior == (MATRIX *) NULL)
            class_results_with_ties[classes[i]] ++;
        else 
            class_results_with_ties[classes[i]] +=
                 (1.0 / *SUB (prior, classes[i], classes[i]));
        i++;
    }

/* 
** Now we're effectively using i-nn, not just k-nn. That's reflected
** in the entries in class_results_with_ties.
*/

/* Now we want to find the maximum. In case of a tie, we use...*/
    tie = 0;
    max_class = -1;
    max_count = -1;
/* 
** For each class, see how many entries in the "classes" array have 
** that number.
*/
    for (i = 0; i < number_of_classes; i++)
    {
/*
** "Results[i]," then, is the number of times "i" appears in "classes." If 
** this is smaller than max_count, move on. If it's equal, note that a tie 
** exists.  Otherwise, save the count and the number of this class.
*/
        if (class_results_with_ties[i] < max_count)
            continue;
        if (class_results_with_ties[i] == max_count)
        {
            tie = 1;
            continue;
        }
        tie = 0;
        max_class = i;
        max_count = class_results_with_ties[i];
    }

    if (tie == 0)
    {
if (verbose > 2)
    Rprintf ("Max class was %li, putting that in outcome %li\n",
    max_class, k_ctr);

        outcome[k_ctr] = max_class;
        continue;
    }

if (verbose > 2)
    Rprintf ("Got a tie.\n");
/* Make a note of all tied classes.... */
    for (i = 0; i < number_of_classes; i ++)
    {
        if (class_results_with_ties[i] == max_count)
            tie_marker[i] = 1;
    }
/* ...and return the first class that belongs to one of the tied ones. */
    for (i = 0; i < k_ctr; i++)
        if (tie_marker[classes[i]] == 1)
            outcome[k_ctr] = classes[i];

/* We should never get here. */
    outcome[k_ctr] = max_class;

} /* end "for k_ctr" counting through the k's. */

} /* end "xpoll" */

/*=========================  c_euclidean  ==================================*/

double c_euclidean (double *vec_1, double *vec_2, double *c, 
                    long n, double threshold)
{
long i;
double sum;

sum = 0.0;
for (i = 0; i < n; i++)
{
    if (c[i] == 0)
        continue;
    sum += c[i] * (vec_1[i] - vec_2[i]) * (vec_1[i] - vec_2[i]);
    if (threshold > 0 && sum > threshold)
        return (-1.0);
}
return (sum);

} /* end "euclidean" */

/*=========================  f_euclidean  ==================================*/
double f_euclidean (double *vec_1, double *vec_2, double *phi, double *c, 
                    long n, double threshold, Slong *cats_in_var, 
                    Slong *cum_cats_this_subset,
                    MATRIX *prior, double **knots)
{
long col_ctr, knot_ctr, offset;
double temp_1, temp_2;
double obs_1_minus_knot, obs_2_minus_knot;
double sum;
long inc_type = UNORDERED;

/*
** The story goes like this. We want to find not the distance between
** the two vectors vec_1 and vec_2, but rather the distance between
** what you get when you replace all the entries with the relevant
** entries of phi. For a categorical variable, we look at the starting
** point for this variable (in "cum_cats_this_subset"), then look into
** phi at (that point + the value in vec_1 for this variable) to get
** the relevant value. For a numeric variable, we add up the sum
** of (this value - knot) * the relevant phi whenever the first
** term is positive.
**
** One twist: if phi's NULL, just find the distance between the two vecs.
*/

sum = 0.0;
for (col_ctr = 0; col_ctr < n; col_ctr++)
{
    if (c[col_ctr] == 0.0)
        continue;
    if (phi == (double *) NULL)
    {
        offset = 0;
        inc_type = UNORDERED;
    }
    else
    {
        offset = cum_cats_this_subset[col_ctr];
        inc_type = increase[col_ctr];
    }
    temp_1 = 0.0;  temp_2 = 0.0;
    if (inc_type == NUMERIC)
    {
        for (knot_ctr = 0; knot_ctr < cats_in_var[col_ctr]; knot_ctr++)
        {
            obs_1_minus_knot = vec_1[col_ctr] - knots[col_ctr][knot_ctr];
            obs_2_minus_knot = vec_2[col_ctr] - knots[col_ctr][knot_ctr];
            if (obs_1_minus_knot < 0.0
            &&  obs_2_minus_knot < 0.0)
                break;
            if (obs_1_minus_knot > 0.0)
                temp_1 += obs_1_minus_knot * phi[knot_ctr + offset];
            if (obs_2_minus_knot > 0.0)
                temp_2 += obs_2_minus_knot * phi[knot_ctr + offset];
        }
    }
    else
    {
        if (phi == (double *) NULL)
        {
            temp_1 = vec_1[col_ctr];
            temp_2 = vec_2[col_ctr];
        }
        else
        {
            temp_1 = phi[( (long) vec_1[col_ctr]) + offset];
            temp_2 = phi[( (long) vec_2[col_ctr]) + offset];
        }
    }

    sum += c[col_ctr] * (temp_1 - temp_2) * (temp_1 - temp_2);

if (verbose > 4)
{
    Rprintf ("Col %i c %f; t1 %f (%i) t2 %f (%i), so sum is %f\n",
     col_ctr, c[col_ctr], 
         temp_1, (int) (vec_1[col_ctr] + offset), 
         temp_2, (int) (vec_2[col_ctr] + offset), sum); 
}

    if (threshold > 0 && sum > threshold)
    {
        if (verbose > 4)
            Rprintf ("Threshold is %f, which is > 0, and sum is %f\n",
                threshold, sum);
        return (-1.0);
    }
} /* end "for" loop */

return (sum);

} /* end "f_euclidean" */

/*=========================  c_absolute  ==================================*/
double c_absolute (double *vec_1, double *vec_2, double *c, 
                   long n, double threshold)
{
long i;
double sum;

sum = 0.0;
for (i = 0; i < n; i++)
{
    sum += c[i] * fabs (vec_1[i] - vec_2[i]);
    if (threshold > 0 && sum > threshold)
        return (-1.0);
}
return (sum);

} /* end "c_absolute" */
