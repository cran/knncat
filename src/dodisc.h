int do_discriminant (MATRIX *test,
              MATRIX *eigenvalues, MATRIX *eigenvectors,
              MATRIX *make_phi, Slong *cats_in_var, Slong *cum_cats,
              long dimension, long number_of_variables, 
              double **knots, MATRIX *cost, MATRIX *prior,
              double *error_rate, MATRIX *misclass_mat, int do_the_omission);
