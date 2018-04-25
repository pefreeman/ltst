# ltst
Code for carrying out local two-sample testing with random forest (see Freeman et al. 2017). The code is in R, although in theory a Python version will be created as well. For astronomers not conversant in R, it is quite easy to get started:
1) Download and install R from [here](https://www.r-project.org)
2) Download and install R Studio from [here](https://www.rstudio.com)
There are any number of R tutorials on the web. An oft-used one is
[simpleR](https://cran.r-project.org/doc/contrib/Verzani-SimpleR.pdf) by John Verzani; one appendix of it in particular covers reading ASCII data into the R environment. If you are dealing with FITS binary table files, contact me (pfreeman at cmu dot edu) and I can provide code for reading them in.

DM.Rmd is an [R markdown](http://rmarkdown.rstudio.com) file that contains code and instructions for constructing a diffusion map given input predictor data. A R markdown file is similar to a Jupyter notebook; it allows for the creation of plots or tabulated output, etc., on a code chunk-by-code chunk basis. For an introduction, see [this page](http://rmarkdown.rstudio.com/articles_intro.html).

RF_TST.Rmd is an R markdown file that contains code and instructions for performing local two-sample hypothesis tests in the manner of Freeman et al. (2017), given input predictor and response data. Our implementation of local two-sample testing involves the use of a version of the random forest algorithm that is based on the work of [Wager et al. (2014)](http://jmlr.org/papers/v15/wager14a.html) and [Wager & Athey (2015)](https://arxiv.org/abs/1510.04342).

Example.Rmd is an R markdown file that takes the data in statistics_iband.Rdata and (a) creates a diffusion map and (b) performs a random forest analysis. The data are those used in Freeman et al. 2017.
