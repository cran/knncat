#ifdef CALL_FROM_R
#define Slong int
#else
#define Slong long
#endif

int do_nn (Slong *quit, MATRIX *training, MATRIX *test, 
           MATRIX *c, Slong *k, long *in_how_many_ks, 
           Slong *theyre_the_same, double *phi, Slong *cats_in_var, 
           Slong *cum_cats_this_subset,
           double **knots, MATRIX *cost, 
           Slong *prior_ind, MATRIX *prior, double *error_rates,
           MATRIX *misclass_mat, Slong *return_classes, Slong *classes,
           long *in_xval_lower, long *in_xval_upper, long *in_xval_indices,
           Slong *in_increase, Slong *in_number_of_classes, Slong *in_verbose);
