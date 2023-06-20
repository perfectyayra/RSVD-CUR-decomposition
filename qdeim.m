function irow = qdeim(U,k)

%Index selection based on Q-DEIM
% U is singular vectors
% k desired number of indices
%Reference: Gugercin and Drmac, 2015

[~,~,p] = qr(U','vector') ; irow = p(1:k) ;
