n=100000;
m=1000;
k=50;
C=ones(m) + diag(4*diag(ones(m))); %compound symmetry covariance structure 
C1=toeplitz(0.99.^(0:n-1));  %auto regressive of order 1 covariance structure
R=chol(C);
R1=chol(C1);
A_exact=zeros(m,n);


for i=1:10
    x=sprand(m, 1,0.025);
    y=sprand(n, 1,0.025);
    A_exact=A_exact+(2/i)*x*y.';
end 
for i=11:100
    x=sprand(m, 1,0.025);
    y=sprand(n, 1,0.025);
    A_exact=A_exact+(1/i)*x*y.';
end 


for j=1:10
    ee= R*randn(m,n)*R1 ; %correlated noise 
    E=0.1*(norm(A_exact)/norm(ee))*ee;
    A=A_exact+E;
    [U,S,V]=svd(A,0); %matlab implementation
    [Z,W,U1,V1,SA,SB,SC] = rsvd(A,R,R1); %our implementation 

    for i=1:k
        %% DEIM-CUR 
        icol = deim(V(:,1:i),i);
        irow = deim(U(:,1:i),i);
        M=A(:,icol(1:i))\A/A(irow(1:i),:);
        CUR=A(:,icol(1:i))*M*A(irow(1:i),:);
        
        %% DEIM-RSVD_CUR
        icol1 = deim(W(:,1:i),i);
        irow1 = deim(Z(:,1:i),i);
        M1=A(:,icol1(1:i))\A/A(irow1(1:i),:);
        RCUR=A(:,icol1(1:i))*M1*A(irow1(1:i),:);
        
        %% Truncated SVD and RSVD
        A_svd=U(:,1:i)*S(1:i,1:i)*V(:,1:i)';
        A_rsvd=Z(:,1:i)*SA(1:i,1:i)*W(:,1:i)';
        
        %% errors
        CUR_err(j,i)=norm(A_exact-CUR)/norm(A_exact);
        RCUR_err1(j,i)=norm(A_exact-RCUR)/norm(A_exact);
        SVD_err(j,i)=norm(A_exact-A_svd)/norm(A_exact);
        RSVD_err(j,i)=norm(A_exact-A_rsvd)/norm(A_exact);

    end
end

err=mean(CUR_err);
err1=mean(RCUR_err);
err2=mean(SVD_err);
err3=mean(RSVD_err);



plot(1:50,err,'r-');
hold on;
plot(1:50,err1,'b-'); 
plot(1:50,err2,'r-.');
plot(1:50,err3,'b-.');
  
legend('DEIM-CUR','DEIM-RCUR','SVD','RSVD')
