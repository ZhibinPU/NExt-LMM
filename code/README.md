## NExt-LMM.py

This module contains two class: the HODLR Inverse and NExt-LMM.
*class* **HInverse**(X = None, block_min = 3)
+ *class* **NExt**(Y = None, S_inv = [])

### Parameters
Some of the parameters are significant in the class construction.

+ **X: array-like of shape (n, n), default = None**: a $n \times n$ GSM-based covariance matrix, n is the number of individuals.
+ **Y: array-like of shape (n, 1), default = None**: a $n \times 1$ phenotype vector, n is number of individuals.
+ **block_min: int, default = 3**: the user-specific minimum block size in the HoDLR structure.
+ **S_inv: array-like of shape (n, n), default = array-like of shape (0, 0)**: the inversion of X if pre-calculated. 

Some of the parameters are used in the association testing function.
+ **W: array-like of shape (n, d), default = None**: a $n \times d$ genotype matrix, n is the number of individuals and d is the number of genotypes.
+ **K: array-like of shape (n, n), default = None**: the $n \times n$ GSM, n is the number of individuals.
+ **h: int, default = None**: the SNP-based heritability if given.

### Output
+ **sigma: int**: the estimation of the variance component.
+ **beta: array-like of shape (1, d)**: the estimation of effect sizes.
+ **p: array-like of shape (1, d)**: P-values of the estimated effect sizes.


## Demo.ipynb

The file *demo.ipynb* consists of the following three steps:

+ Initialize the hyperparameters: a = c = 2, b = d = 1, mu = np.array([0, 0]), G = np.array([[1, 0.5], [0.5, 1]]) and the number of MCMC iterations = 5000.
+ Apply the linear mixed model to the pseudo-snps in *snp.txt* and corresponding traits in *trait.txt*, to obtain the coefficient $\beta$.
+ Draw trace plots and histograms of $\beta$.

### Notes:
For a single genotype, and a $2 \times 100$ phenotype matrix (i.e. 100 individuals with 2 QTs per individual), it would take approximately *<10 seconds* to  obtain a 5000-iterations MC chain when implemented in the following computational environment:


**CPU:** 3.20-GHz AMD Ryzen 7 7735H; **Memory**: 16 GB RAM

with following package versions.
+ **pandas**: 1.4.4
+ **numpy**: 1.21.6
+ **scipy**: 1.9.1
+ **matplotlib**: 3.5.2
+ **tqdm**: 4.64.1
