function irow = qdeim(U,k)
%Index selection based on Q-DEIM

[~,~,p] = qr(U','vector') ; irow = p(1:k) ;
