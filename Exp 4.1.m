n=100000;
m=1000;
k=50;
C=ones(m) + diag(4*diag(ones(m))); %compound symmetry covariance structure 
C1=toeplitz(0.99.^(0:n-1));  %auto regressive of order 1 covariance structure
B=chol(C);
G=chol(C1);
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
    ee= B*randn(m,n)*G ; %correlated noise 
    E=0.1*(norm(A_exact)/norm(ee))*ee;
    A=A_exact+E;
    [U,S,V]=svd(A,0); %matlab implementation
    [Z,W,U1,V1,SA,SB,SC] = rsvd(A,B,G); %our implementation 

    for i=1:k
        %% DEIM-CUR 
        icol = deim(V(:,1:i),i);
        irow = deim(U(:,1:i),i);
        M=A(:,icol(1:i))\A/A(irow(1:i),:);
        CUR=A(:,icol(1:i))*M*A(irow(1:i),:);
        
        %% QDEIM-CUR 
        icol1 = qdeim(V(:,1:i),i);
        irow1 = qdeim(U(:,1:i),i);
        M1=A(:,icol1(1:i))\A/A(irow1(1:i),:);
        CUR1=A(:,icol1(1:i))*M1*A(irow1(1:i),:);
        
        %% DEIM-RSVD_CUR
        icol11 = deim(W(:,1:i),i);
        irow11 = deim(Z(:,1:i),i);
        M11=A(:,icol11(1:i))\A/A(irow11(1:i),:);
        RCUR=A(:,icol11(1:i))*M11*A(irow11(1:i),:);
        
        %% QDEIM-RSVD_CUR
        icol12 = qdeim(W(:,1:i),i);
        irow12 = qdeim(Z(:,1:i),i);
        M12=A(:,icol12(1:i))\A/A(irow12(1:i),:);
        RCUR1=A(:,icol12(1:i))*M12*A(irow12(1:i),:);
        
        %% Truncated SVD and RSVD
        A_svd=U(:,1:i)*S(1:i,1:i)*V(:,1:i)';
        A_rsvd=Z(:,1:i)*SA(1:i,1:i)*W(:,1:i)';
        
        %% errors
        CUR_err(j,i)=norm(A_exact-CUR)/norm(A_exact);
        CUR1_err(j,i)=norm(A_exact-CUR1)/norm(A_exact);
        RCUR_err1(j,i)=norm(A_exact-RCUR)/norm(A_exact);
        RCUR1_err1(j,i)=norm(A_exact-RCUR1)/norm(A_exact);
        SVD_err(j,i)=norm(A_exact-A_svd)/norm(A_exact);
        RSVD_err(j,i)=norm(A_exact-A_rsvd)/norm(A_exact);

    end
end

err=mean(CUR_err);
err1=mean(CUR1_err);
err11=mean(RCUR_err);
err12=mean(RCUR1_err);
err2=mean(SVD_err);
err3=mean(RSVD_err);



plot(4:2:k,nonzeros(err),'-o');
hold on;
plot(4:2:k,nonzeros(err1),'-*'); 
plot(4:2:k,nonzeros(err11),'-s'); 
plot(4:2:k,nonzeros(err12),'-p'); 
plot(4:2:k,nonzeros(err2),'-d');  
plot(4:2:k,nonzeros(err3),'-+');  

legend('DEIM-CUR','QDEIM-CUR','DEIM-RSVD-CUR','QDEIM-RSVD-CUR','SVD','RSVD')
