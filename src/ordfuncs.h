/*
** Ordfuncs.h: header file for prototypes of ords functions.
**
** Down at the bottom are the prototypes for the "outside"
** (NAG, LAPACK) routines.
**
*/
#ifndef CALL_FROM_S
int get_options (int argc, char **argv);
int count_fields (char *buf);
int convert_input (char *buf, double *in_vec, long n);
#endif

void matrix_element_divide (MATRIX *result, MATRIX *num, MATRIX *denom);

int fill_margin_holder (MATRIX *margin_holder, long *permute_index, 
                        long which,
                        MATRIX *data, long *cats_in_var, 
                        double **knots, long *cum_cats_this_subset, 
                        long *number_in_class, long *increase);

void constraint (long *mode,  long *how_many, long *vars, long *nrow,
                 double *x, double *value, double *jacobian, long *nstate);

void objective (long *mode, long *n, double *x, double *objvalue,
                double *gradient, long *nstate);

int fill_u (MATRIX *training,
            long *cats_in_var, long *cum_cats_this_subset,
            long *number_in_class, MATRIX *margin_ptr);

int fill_row_of_u_submatrices (MATRIX *training,
    long starting_row, long *cats_in_var,
    long *cum_cats_this_subset, long *number_in_class, MATRIX *margin_ptr);

int fill_w (MATRIX *training,
        double **knots, long *cats_in_var, long *cum_cats_this_subset, 
        MATRIX *margin_ptr, long i, 
        long *permute_indices, long *increase);

int fill_row_of_w_submatrices (MATRIX *training,
    long starting_row, double **knots, long *cats_in_var,
    long *cum_cats_this_subset, MATRIX *margin_ptr, long which_to_permute,
    long *permute_indices, long *increase);

int divide_by_root_before_and_after (MATRIX *in, MATRIX *diag);

int divide_by_root_prior (MATRIX *in, MATRIX *prior);

int multiply_by_root_prior (MATRIX *in, MATRIX *prior);

int resize_matrix (MATRIX **a, long nrow, long ncol);

int adjust_column (unsigned long column, MATRIX *adjustee, long increase,
                   double **knots, long cats_in_this_var, MATRIX *phi);

int handle_fascinating_file (FILE *fascinating_file, int *are_any_numeric);

double add_ordered_variable (double *result, long which_var, long dimension,
                      long *cats_in_var,
                      MATRIX *eigenvectors, long *increase);

void deal_with_missing_values (MATRIX *training, double missing_max, 
                              long *increase, long *cats_in_var, 
                              double *missing_values);

int insert_missing_values (MATRIX *mat, double missing_max, 
                           double *missing_values);

int do_the_eigen_thing (MATRIX *eigenval_ptr, 
                    MATRIX *eigenvec_ptr, double ridge,
                    MATRIX *eigenvalues_imaginary, MATRIX *eigenvalues_beta);

int get_a_solution (long permute_ctr, long *permute_indices, 
                    long permute_len, long i, MATRIX *margin_ptr, 
                    MATRIX *training, 
                    long *cats_in_var, long *cum_cats_ptr, double **knots,
                    long *number_in_class, MATRIX *eigenval_ptr, 
                    MATRIX *eigenvec_ptr, MATRIX *w_inv_m, 
                    double ridge, int classification, 
                    long number_of_vars, int dimension, long current_cat_total,
                    MATRIX **prior, int prior_ind, int prior_len, 
                    int first_time_through, int do_the_omission, 
                    long *increase, int quit);

int prepare_eigen_matrices (
                 MATRIX **original_eigenvalues, MATRIX **original_eigenvectors, 
                 MATRIX **original_w_inv_m_mat, MATRIX **best_w_inv_m,
                 MATRIX **eigenvalues_real,     MATRIX **eigenvectors,
                 MATRIX **best_eigenvector,
                 long classification, long first_time_through,
                 long number_of_classes, long number_of_cats);

void count_them_cats (long number_of_vars,
                     long *current_cat_total, long *currently_ordered, 
                     long *currently_numeric,
                     long *orderable_cats, MATRIX *c, long *increase);

int get_sequence_of_solutions (long quit_dimension,
    long number_of_vars, long permute, long permute_len, long *permute_indices,
    double improvement, long k_len, long *k,
    long *cats_in_var, long *cum_cats_ptr, MATRIX *c,
    MATRIX **original_eigenvectors, MATRIX **original_w_inv_m_mat, 
    long *number_in_class, 
    long xval_ctr,
    MATRIX *xval_result, long *xval_ceiling,
    MATRIX *cost, MATRIX *prior, long prior_ind, long number_of_classes,
    double *misclass_rate, int do_the_omission, long *increase,
    long *once_out_always_out);

int do_the_discriminant_thing (long permute_ctr,
       MATRIX *eigenval_ptr, MATRIX *eigenvec_ptr, MATRIX *original_w_inv_m_mat,
       MATRIX *prior, long i, 
       long number_of_vars, int dimension, 
       long current_cat_total, int do_the_omission, long *increase);


/* Nag routines for eigenvalues. */
void f02abf ();
void f02aef ();

/* NAG routine to sort a numeric vector */
void m01anf_(double *, long *, long *, long *);

/* NAG routines to do non-linear optimization */
void e04zcf_ ();
void e04ucf_ (long *current_cat_total,  /* number of vars */
              long *lin_const_count,    /* number of linear constraints */
              long *nonlin_const_count, /* number of nonlin constraints */
              long *lda,                /* number of rows in A matrix */
              long *ldcj,               /* number of rows in Jacobian */
              long *ldr,                /* number of rows in R           */
              double *a,                /* matrix of linear constraints */
              double *lower_bounds, double *upper_bounds, /* bounds */
              void (*confun)(), void (*objfun)(),
              long *e04_iter, long *istate,
double *c, double *jacobian_result, double *e04_lambda,
double *objective_result, double *gradient_result,
double *R, double *X,
long *long_work, long *length_long_work,
double  *double_work, long *length_double_work,
long *iuser, double *user,
long *NAG_status);

/* NAG routines to do non-linear optimization */
void e04vdf_ (long *e04_iter, long *e04_msglvl, long *current_cat_total, 
              long *lin_const_count, long *nonlin_const_count,
              long *nc_total, long *lin_const_count2, long *nonlin_const_count2,
              double *ctol, double *ftol, double *constraint_matrix, 
              double *lower_bounds, double *upper_bounds, void (*confun)(),
              void (*objfun)(), double *phi_data, long *istate, 
              double *nonlin_const_value, double *jacobian_result,
              double *objective_result, double *gradient_result,
              double *e04_lambda, long *long_work, long *length_long_work,
              double  *double_work, long *length_double_work, long *NAG_status);

void e04vcf_ (long *e04_iter, long *e04_msglvl, long *current_cat_total, 
              long *lin_const_count, long *nonlin_const_count,
              long *nc_total, long *lin_const_count2, long *nonlin_const_count2,
              long *nrow_R, double *bigbnd, double *epsaf, double *eta,
              double *ftol, double *constraint_matrix, 
              double *lower_bounds, double *upper_bounds, 
              double *featol, void (*confun)(), void (*objfun)(), 
              int *cold, int *fealin, int *orthog,
              double *phi_data, long *istate, 
              double *R, long *iter_result, 
              double *nonlin_const_value, double *jacobian_result,
              double *objective_result, double *gradient_result,
              double *e04_lambda, long *long_work, long *length_long_work,
              double  *double_work, long *length_double_work, long *NAG_status);

/* NAG routine to do permutation of integers */
void g05ehf_(long *, long *, long *);
/* NAG routine to set up random generator */
void g05ccf_();

long NAG_status;


/* EISPACK routines for eigenvalues. */
void rgg_(unsigned long *, unsigned long *, double *, double *,
double *, double *, double *, long *, double *, long *);
void rsg_();

/* LAPACK routine for sorting */
void dsort_ (double *data, double *other_data, long *length, long *which);

/* Ranlib routine to do permutation of integers */
void genprm (long *, int);
