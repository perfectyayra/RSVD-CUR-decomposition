# A restricted SVD based CUR decomposition for matrix triplets

### Authors: Perfect Y. Gidisu and Michiel E. Hochstenbach

The MATLAB code provided in the various files implements the algorithms described in the paper titled "A restricted SVD based CUR decomposition for matrix triplets." For a comprehensive understanding of these algorithms, we recommend referring to the paper available at https://arxiv.org/abs/2204.02113.

The experiment files, named Exp_4_1.m, Exp_4_2.m, and so on, correspond to the experiment numbers mentioned in the paper. Each of these files includes the necessary functions and code required to reproduce the experiments conducted in the study. These files serve as a guide for replicating the experimental setup. The names of the columns in the train and test data of the Exp_4_4 are picked from the paper "Razia, S., SwathiPrathyusha, P., Vamsi Krishna, N., & Sathya Sumana, N. (2018). A Comparative study of machine learning algorithms on thyroid disease prediction. International Journal of Engineering & Technology, 7(2.8), 315-319. https://doi.org/10.14419/ijet.v7i2.8.10432".
##### NB: In the exception of experiment Exp_4_3, in all other experiments random cases are taken. Therefore, depending on the random seeds, the results may be slightly different from the reported values. However, this should not change the essence and conclusions from the experiments.

Here is a brief description of the files included in the repository:

rsvd.m: This file implements Algorithm 2.1, which is discussed in detail in the paper. It contains the code necessary to perform a restricted singular value decomposition (RSVD) of matrices.

deim.m: This file contains the implementation of Algorithm 3.1, an essential component of the overall algorithm. Algorithm 3.1 describes the discrete empirical interpolation method (DEIM) utilized in the process.

qdeim.m: Here, a variant of the DEIM scheme called the pivoted QR factorization variant is implemented. This file provides the code for this specific DEIM variant.

rsvd_cur_deim.m: This file implements Algorithm 3.2, which combines the previously mentioned algorithms (RSVD, DEIM) to perform the RSVD-CUR decomposition. The code in this file showcases how the algorithms are integrated and applied to the matrix triplets.

smin.m: This function is utilized within the code to reproduce Figure 4.2 in the paper. 

Additionally, the code utilizes specific data sets for conducting experiments. The mfeat-fou, mfeat-kar, and mfeat-pix data sets are used in Experiment 4.3. On the other hand, the ann-train.data and ann-test.data data sets are used in Experiment 4.4.

By utilizing these code files, functions, and provided data sets, it is possible to replicate the experiments and obtain the results presented in the paper. These resources serve as a valuable tool for understanding and applying the restricted SVD-based CUR decomposition technique to matrix triplets.
