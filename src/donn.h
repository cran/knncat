#ifdef CALL_FROM_R
#define Slong int
#else
#define Slong long
#endif

int do_nn (long *quit, 
              MATRIX *training, MATRIX *test,     
              MATRIX *c, long *k, long *how_many_ks, long *theyre_the_same, 
              double *phi, long *cats_in_var, long *cum_cats_ptr,
              double **knots, MATRIX *cost, MATRIX *prior,
              double *error_rates, MATRIX *misclass_mat,
              long *return_classes, Slong *classes,
              long *xval_lower, long *xval_upper, long *xval_indices,
              long *increase, long *number_of_classes, long *verbose);

