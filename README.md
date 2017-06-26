# ltst
Code for carrying out local two-sample testing with random forest (see Freeman et al. 2017).

DM.Rmd is an R markdown file that contains code and instructions for constructing a diffusion map given input predictor data.

RF_TST.Rmd is an R markdown file that contains code and instructions for performing local two-sample hypothesis tests in the
  manner of Freeman et al. (2017), given input predictor and response data. Our implementation of local two-sample testing 
  involves the use of a version of the random forest algorithm that is based on the work of [Wager et al. (2014)](http://jmlr.org/papers/v15/wager14a.html) and [Wager & Athey (2015)](https://arxiv.org/abs/1510.04342).
