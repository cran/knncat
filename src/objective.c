/*
**         Objective.c
**
** This file holds the functions that compute the objective function
** (and its gradient) and the constraint function (and its jacobian)
** to be used in the non-linear optimization routine (after an
** ordered variable has entered the model). Constraint() is called
** first; we do all the computations in there (because as it happens
** the objective is part of the constraint, anyway), and save them globally.
** Then when objective() is called we can just trot 'em out.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include "matrix.h"
#include "utils.h"
#include "ord.h"
#include "R.h"

#ifdef USE_R_ALLOC
extern char *R_alloc();
void do_nothing();
#define free do_nothing
#define calloc R_alloc
#endif
#ifdef CALL_FROM_R
#define Slong int
#else
#define Slong long
#endif

MATRIX global_gradient;
double global_objective_value;
extern MATRIX *global_copy_of_w, *global_copy_of_u;
extern Slong verbose;

/*=========================== constraint ===============================*/
void constraint (long *mode,  long *how_many, long *vars, long *nrow,
             double *x, double *value, double *jacobian, long *nstate)
{
static MATRIX phi;
static MATRIX jacobian_mat, jacobian_hold;
static MATRIX result_mat;
double result;
static long been_called_already = FALSE;

if (*nstate == 1)
{
    phi.nrow = *vars; 
    phi.ncol = 1L;
    strcpy (phi.name, "Phi");

    jacobian_mat.nrow = jacobian_hold.nrow = *vars; 
    jacobian_mat.ncol = jacobian_hold.ncol = 1L;
    strcpy (jacobian_mat.name,      "Jacobian");
    strcpy (jacobian_hold.name, "Jacobian Hold");

    global_gradient.nrow = *vars; 
    global_gradient.ncol = 1L;
    strcpy (global_gradient.name, "Gradient");

    result_mat.nrow = 1L;
    result_mat.ncol = 1L;
    strcpy (result_mat.name, "Result");
    result_mat.data = &result;

    if (been_called_already == FALSE)
    {
        been_called_already = TRUE;
    }
    else
    {
/****
        jacobian_hold.data = (double *) realloc ((char *) &(jacobian_hold.data),
                                 (unsigned) (*vars * sizeof (double)));
        global_gradient.data = (double *) realloc ((char *) 
                              &(global_gradient.data), *vars * sizeof (double));
****/
        free (jacobian_hold.data);
        free (global_gradient.data);
    }
    alloc_some_doubles (&(jacobian_hold.data), *vars);
    alloc_some_doubles (&(global_gradient.data), *vars);
}
phi.data          = x;
jacobian_mat.data = jacobian;
zero_matrix (&global_gradient);
zero_matrix (&jacobian_mat);

matrix_multiply (global_copy_of_w, &phi, &global_gradient, NO_TRANSPOSES);
matrix_multiply (&phi, &global_gradient, &result_mat, TRANSPOSE_FIRST);

global_objective_value = - result_mat.data[0];
value[0] = result_mat.data[0];

matrix_multiply (global_copy_of_u, &phi, &jacobian_hold, NO_TRANSPOSES);
matrix_multiply (&phi, &jacobian_hold, &result_mat, TRANSPOSE_FIRST);

value[0] -= result_mat.data[0];

matrix_add (&global_gradient, &jacobian_hold, &jacobian_mat, SUBTRACT);
scalar_multiply (&global_gradient, (MATRIX *) NULL, -2.0);
scalar_multiply (&jacobian_mat, (MATRIX *) NULL, 2.0);

if (verbose > 0)
{
    print_matrix (&phi, TRANSPOSE_FIRST);
    print_matrix (&jacobian_mat, TRANSPOSE_FIRST);
    Rprintf ("Constraint is %f\n", value[0]);
}


} /* end "constraint" */

/****************************** objective *****************************/
void objective (long *mode, long *n, double *x, double *objvalue,
                double *gradient, long *nstate)
{
long i;

for (i = 0L; i < global_gradient.nrow; i++)
    gradient[i] = global_gradient.data[i];

*objvalue = global_objective_value;

if (verbose > 0)
    Rprintf ("Hi. Objective is %f\n", *objvalue);

} /* end "objective" */



