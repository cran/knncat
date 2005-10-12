/*
**      do_discriminant ()
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "utils.h"
#include "ord.h"

#ifdef USE_R_ALLOC
void do_nothing();
#define free do_nothing
#endif

extern int verbose;
extern long *xval_indices;
extern long xval_lower;
extern long xval_upper;

extern long *am_i_in;
extern long *increase;

double sum_of_phis (long, double *, Slong *, Slong *, double **, double *);
int    expand_vector (MATRIX *, MATRIX *, long, long,
                      Slong *, double *);

/*=========================== do_discriminant ===========================*/

int do_discriminant (MATRIX *test, MATRIX *eigenvalues, MATRIX *eigenvectors,
              MATRIX *make_phi, Slong *cats_in_var, Slong *cum_cats_this_subset,
              long dimension, long number_of_variables,
              double **knots, MATRIX *cost, MATRIX *prior, double *error_rate,
              MATRIX *misclass_mat, int do_the_omission)
{
MATRIX *short_phi, *short_phi_transposed, *phi;
MATRIX *short_phi_row_ptr, *phi_row_ptr;
unsigned long number_of_classes;
unsigned long cat_total, cat_total_short;
unsigned long test_ctr;
unsigned long class_ctr, l_ctr;
long test_class; /* Class of test item */
long test_item_count; /* Actual number of test items considered. */
long shortest_class;
double add_dist, dist, shortest_dist, my_sum;
double this_cost;
unsigned long misclass;
double *class_dist;
double *one_minus_eigen;
long first_solution, last_solution;
int status;

cat_total_short = make_phi->nrow;
/*
** If "do_the_omission" is false, cat_total and cat_total short are alike. 
*/
cat_total = do_the_omission? cat_total_short + dimension - 1 : cat_total_short;
number_of_classes = eigenvalues->nrow;
alloc_some_doubles (&class_dist, number_of_classes);

/* "Short_phi" is "short" because variables 2 through however many
** there are have had the first category deleted. So we create short_phi,
** by multiplying the "make_phi" matrix by one of the eigenvectors
** in "eigenvectors," and then expand it by calling "expand_vector."
** At the end of that operation, we have the full phi for one of
** the eigenvectors. We do that for all the eigenvectors simultaneously,
** and then we compute the distance from the sum of phis for each 
** observation to the corresponding theta...more later.
*/
short_phi_transposed = make_matrix (cat_total_short, number_of_classes, 
                             "Short Phi T", REGULAR, TRUE);
short_phi            = make_matrix (number_of_classes,  cat_total_short,
                             "Short Phi", REGULAR, TRUE);
short_phi_row_ptr    = make_matrix ((long) 1,  cat_total_short,
                             "Short Phi Row", REGULAR, FALSE);
phi                  = make_matrix (number_of_classes, cat_total, 
                              "Phi", REGULAR, ZERO_THE_MATRIX);
phi_row_ptr          = make_matrix ((long) 1, cat_total, 
                              "Phi row", REGULAR, FALSE);

/***
matrix_multiply (make_phi, eigenvectors, short_phi_transposed, NO_TRANSPOSES);
***/
matrix_multiply (make_phi, eigenvectors, short_phi_transposed, 
                 TRANSPOSE_SECOND);
matrix_copy_transpose (short_phi, short_phi_transposed);
free_matrix (short_phi_transposed);


if (dimension == 1 || do_the_omission == FALSE)
{
    matrix_copy (phi, short_phi);
}
else
{
    for (class_ctr = 0; class_ctr < number_of_classes; class_ctr++)
    {
        short_phi_row_ptr->data = SUB (short_phi, class_ctr, 0L);
        phi_row_ptr->data       = SUB (phi, class_ctr, 0L);
        status = expand_vector (phi_row_ptr, short_phi_row_ptr, dimension - 1,
                       number_of_variables, cum_cats_this_subset,
                       (double *) NULL);
    }
}

/* There used to be "number_of_classes - 1" of these. Bear with me. */
alloc_some_doubles (&one_minus_eigen, number_of_classes);

for (class_ctr = 0; class_ctr < number_of_classes; class_ctr++)
    one_minus_eigen[class_ctr] = 1.0 - eigenvalues->data[class_ctr];

misclass = 0L;
test_item_count = 0L;

/*
** Zero out the misclass_matrix, if there is one. 
*/
if (misclass_mat != (MATRIX *) NULL)
    zero_matrix (misclass_mat);

/* Here's how many solution we'll consider: min (number_of_classes - 2, 
** cat_total). The first is "-2" because number_of_classes is one-based.
*/

last_solution  = number_of_classes - 2;
if (cat_total > number_of_classes - 1)
    first_solution = 0;
else
    first_solution = number_of_classes - cat_total;

for (test_ctr = 0; test_ctr < test->nrow; test_ctr++)
{
/* If "xval_indices" isn't NULL, we're probably doing cross-validation.
** Include only items for which test_ctr is bigger than xval_lower
** and less than or equal to xval_upper. Weird case: if xval_lower
** is bigger or equal (as in, if they're both 0), fuggedaboudit.
*/
    if (xval_indices != (long *) NULL)
    {
            if ((xval_lower < xval_upper)
            && (test_ctr < xval_lower || test_ctr >= xval_upper))
                continue;
    }
    test_item_count ++;
    shortest_class = -1;
    shortest_dist  = 0.0;
    for (class_ctr = 0; class_ctr < number_of_classes; class_ctr++)
    {
        dist = 0.0;
        for (l_ctr = first_solution; l_ctr <= last_solution; l_ctr++)
        {
            my_sum = sum_of_phis (number_of_variables, SUB (phi, l_ctr, 0L), 
                                  cats_in_var, cum_cats_this_subset,
                                  knots, SUB (test, test_ctr, 1L));
            add_dist = pow (*SUB (eigenvectors, l_ctr, class_ctr) - my_sum, 2.0)
                       / one_minus_eigen[l_ctr];
/*
** If a cost matrix was passed in, use it. We only handle cost matrices
** with constant rows (that is, the misclass cost depends only on the
** true class). So we extract the value for this class, and column 0:
** that's as good as any, unless this is class 0 (then, use column 1).
** We divide by that cost. THIS ONLY WORKS FOR 2x2 COST MATRICES!
*/
            if (cost == (MATRIX *) NULL)
                dist += add_dist;
            else
            {
                this_cost = *SUB (cost, class_ctr, (class_ctr == 0? 1:0) );
                if (this_cost <= 0)
                {
                    fprintf (stderr, "Illegal cost matrix: abandoning!\n");
                    cost = (MATRIX *) NULL;
                    dist += add_dist;
                }
                else
                    dist += add_dist / this_cost;
            }
/***
            dist += pow (*SUB (eigenvectors, class_ctr, l_ctr) - my_sum, 2.0)
                   / one_minus_eigen[l_ctr];
***/
            if (verbose > 1)
                printf ("Theta is %f, my_sum is %f, divided by %f\n",
                *SUB (eigenvectors, l_ctr, class_ctr), my_sum, 
                    one_minus_eigen[l_ctr]);
            if (verbose >= 2)
                printf ("Dist is...%f\n", dist);

        }
if (verbose > 1)
printf ("  Class ctr %ld, dist %8.8f...\n", class_ctr, dist);
        if (shortest_class < 0
        ||  dist < shortest_dist)
        {
            shortest_dist  = dist;
            shortest_class = class_ctr;
        }
    } /* end "for class_ctr" loop */

    test_class = (long) *SUB (test, test_ctr, 0L);
    if (misclass_mat != (MATRIX *) NULL)
        *SUB (misclass_mat, test_class, shortest_class)
            = *SUB (misclass_mat, test_class, shortest_class) + 1;

    if (shortest_class != test_class)
    {
        if (verbose >= 1)
            printf ("%ld: Misclass a %ld as a %ld\n", test_ctr, 
                  test_class, shortest_class);
            if (cost == (MATRIX *) NULL)
                misclass ++;
            else
                misclass += *SUB (cost, test_class, shortest_class);
    }
if (verbose > 1)
printf ("%ld: Data %ld %ld (%ld) %ld, class %ld assigned to %ld\n", test_ctr, 
              (long) *SUB (test, test_ctr, 1L),
              (long) *SUB (test, test_ctr, 2L),
              (long) *SUB (test, test_ctr, 3L),
              (long) *SUB (test, test_ctr, 4L),
              (long) *SUB (test, test_ctr, 0L), shortest_class);

} /* end "for test_ctr" loop */

*error_rate = (double) misclass / (double) test_item_count;

if (verbose > 0)
{
    print_matrix (eigenvectors,64);
    print_matrix (phi,64);
}

free_matrix (short_phi); free (short_phi_row_ptr);
free_matrix (phi);       free (phi_row_ptr);

return (TRUE);
} /* end "do_discriminate" */

/*=========================== sum_of_phis ==============================*/
double sum_of_phis (long number_of_variables, double *phi,
                    Slong *cats_in_var, Slong *cum_cats,
                    double **knots, double *data)
{
long var_ctr, offset;
double obs_minus_knot, sum;
long knot_ctr;

sum = 0.0;

for (var_ctr = 0; var_ctr < number_of_variables; var_ctr++)
{
    if (am_i_in[var_ctr] <= CURRENTLY_IN)
        continue;
    offset = cum_cats[var_ctr];
    if (increase[var_ctr] == NUMERIC)
    {
        for (knot_ctr = 0; knot_ctr < cats_in_var[var_ctr]; knot_ctr++)
        {
            obs_minus_knot = data[var_ctr] - knots[var_ctr][knot_ctr];
            if (obs_minus_knot <= 0.0)
                break;
            sum += obs_minus_knot * phi[knot_ctr + offset];
        }
    }   
    else
    {
        sum += phi[(long) data[var_ctr] + offset];
    }

}

return (sum);

}

/*=========================== expand_vector ==============================*/
int  expand_vector (MATRIX *new, MATRIX *old, long how_many_holes, 
                   long number_of_variables, Slong *holes,
                   double *filler)
{
/*
** Okay. In the discriminant case we end up with a vector which
** has some "holes." That is, we have deleted some entries to make
** some matrix full rank, and now we want them back. This function
** copies the entries in "old" to "new", except we skip one for
** each element of "holes" and insert the correponding element of
** "filler".  If filler is NULL, we use zeros. So if old is
** 1, 2, 3, 4, 5, 6, and holes is 3, 6, and filler is 11, 12, then
** at the end of this operation, "new" should have
** 1, 2, 11, 3, 4, 5, 12, 6. We allow the special holes 0 (meaning
** insert before anything else) and values bigger than the length
** of "old" (meaning insert them at the end). Holes has to be in
** order.
*/
unsigned long old_ctr, new_ctr, hole_ctr, var_ctr;
unsigned long second_var_in = 0, found_first;

if (new->nrow != 1 || old->nrow != 1)
{
    fprintf (stderr, "Expand vector called, but one has rows not equal 1\n");
    return (FALSE);
}

if (new->ncol != old->ncol + how_many_holes)
{
    fprintf (stderr, "Expand vector called, but sizes are wrong\n");
    return (FALSE);
}

old_ctr = 0;
new_ctr = 0;
hole_ctr = 0;  /* The first entry in "holes" is not of interest. */

found_first = FALSE;
for (var_ctr = 0; var_ctr < number_of_variables; var_ctr++)
{
    if (am_i_in[var_ctr] >= CURRENTLY_IN)
    {
        if (found_first == FALSE)
        {
            found_first = TRUE;
            continue;
        }
        second_var_in = var_ctr;
        break;
    }
}
hole_ctr = second_var_in;

while (new_ctr < new->ncol)
{
    if (holes != (Slong *) NULL && (unsigned long) holes[hole_ctr] == new_ctr)
    {
        if (filler == (double *) NULL)
            new->data[new_ctr] = 0.0;
        else
            new->data[new_ctr] = filler[hole_ctr];
        hole_ctr++;
        while (am_i_in[hole_ctr] < CURRENTLY_IN)
            hole_ctr++;
        new_ctr++;
    }
    else
    {
        new->data[new_ctr] = old->data[old_ctr];
        new_ctr++;
        old_ctr++;
    }
}

return (TRUE);

} /* end "expand_vector" */
