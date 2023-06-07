function [p, s, p_B,s_G, M_A, M_B,M_G] = rsvd_cur_deim(A, B,G, k)

%RSVD-CUR  DEIM incurred CUR decomposition for matrix triplets
% function [p, s, p_B,s_G, M_A, M_B,M_G] = rsvd_cur_deim(A, B,G, k)
% Matrix A and B should have same number of rows
% Matrix A and G should have same number of columns
% B is of full row rank and G is of full column rank
% k is the desired rank of the approximation 
% p contains the column indices of A and G selected
% s contains the row indices of matrix A and B that has been selected
% p_B contains the column indices of matrix B that has been selected
% s_G contains the row indices of matrix G that has been selected


% C_A = A(:,p);  C_B= B(:,p_B); C_G = G(:,p);  R_A = A(s,:);  R_B = B(s,:); R_G=G(s_G,:);
%  M_A, M_B, and M_G are the middle matrix of the rsvd-cur decomposition of A, B, and G, respectively
% See also cur_deim, gcur_deim
%
% Reference: Gidisu and Hochstenbach 2023 


[Z,W,U,V,SA,SB,SG,flag] = rsvd(A,B,G);  %our rsvd implementation


p=deim(W(:,1:k),k);
s=deim(Z(:,1:k),k);
p_B=deim(U(:,1:k),k);
s_G=deim(V(:,1:k),k);



C_A = A(:,p);  C_B = B(:,p_B);  C_G = G(:,p); 
R_A = A(s,:);  R_B = B(s,:);    R_G = G(s_G,:);
M_A = C_A\(A/R_A);  M_B =C_B\(B/R_B); M_G=C_G\(G/R_G);
