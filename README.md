# A restricted SVD based CUR decomposition for matrix triplets

The MATLAB code provided in the various files implements the algorithms described in the paper titled "A restricted SVD based CUR decomposition for matrix triplets." For a comprehensive understanding of these algorithms, we recommend referring to the paper available at https://arxiv.org/abs/2204.02113.

The experiment files, named Exp 4.1.m, Exp 4.2.m, and so on, correspond to the experiment numbers mentioned in the paper. Each of these files includes the necessary functions and code required to reproduce the experiments conducted in the study. These files serve as a guide for replicating the experimental setup and obtaining the reported results.

Here is a brief description of the files included in the repository:

rsrsvd.m: This file implements Algorithm 2.1, which is discussed in detail in the paper. It contains the code necessary to perform a restricted singular value decomposition (RSVD) of matrices.

deim.m: This file contains the implementation of Algorithm 3.1, an essential component of the overall algorithm. Algorithm 3.1 describes the discrete empirical interpolation method (DEIM) utilized in the process.

qdeim.m: Here, a variant of the DEIM scheme called the pivoted QR factorization variant is implemented. This file provides the code for this specific DEIM variant.

rsvd_cur_deim.m: This file implements Algorithm 3.2, which combines the previously mentioned algorithms (RSVD, DEIM) to perform the RSVD-CUR decomposition. The code in this file showcases how the algorithms are integrated and applied to the matrix triplets.

smin.m: This function is utilized within the code to reproduce Figure 4.2 in the paper. 

Additionally, the code utilizes specific data sets for conducting experiments. The mfeat-fou, mfeat-kar, and mfeat-pix data sets are used in Experiment 4.3. On the other hand, the ann-train.data and ann-test.data data sets are used in Experiment 4.4.

By utilizing these files, functions, and data sets, it is possible to reproduce the experiments and results presented in the paper.
