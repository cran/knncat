/*
** Defines for "classification" versus "discimination." We
** also use "-2" for DONT_KNOW (below under "increase")
*/
#define CLASSIFICATION       1
#define DISCRIMINATION       0

/*
** Defines for "am_i_in". This thing holds the rank, in terms of
** increase in lambda, associated with adding each categorical,
** except that there are the following special values. Never
** test " == ALWAYS_IN": test ">= ALWAYS_IN".
*/
#define ALWAYS_IN     (long)  0
#define CURRENTLY_IN  (long) -1
#define CURRENTLY_OUT (long) -2
#define ALWAYS_OUT    (long) -3

/*
** Defines for "increase". This thing tells us whether a particular
** variable is unordered categorical, ordered (increasing or decreasing),
** or numeric.
*/
#define DONT_KNOW     (long) -2
#define UNORDERED     (long) -1
#define DECREASING    (long)  0
#define INCREASING    (long)  1
#define NUMERIC       (long)  -5

/*
** Defines for "fill_margin_holder" 
*/
#define DONT_PERMUTE       -1
#define JUST_GET_CUM_CATS  -2

#define PERMUTATIONS       0L
#define RIDGE             .003
#define IMPROVEMENT       1e-09

/*
** Define for "fill_margin_holder" 
*/
#define DONT_PERMUTE  -1

/*
** Defines for prior probabilities
*/
#define ESTIMATED      1
#define ALL_EQUAL      2
#define SUPPLIED       3
#define IGNORED        4

/* Knots per (numeric) variable */
#define KNOTS  5L

/* Missing_max. Anything smaller than this is presumed to be "missing." */
#define MISSING_MAX -99

/* Number of cross-validations. */
#define XVALS         10L
