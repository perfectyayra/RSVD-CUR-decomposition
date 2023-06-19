n=1000;
m=10000;
k=50;
C=ones(m) + diag(4*diag(ones(m)));
C1=toeplitz(0.99.^(0:n-1));
B=chol(C);
G=chol(C1);
A_exact=zeros(m,n);

for i=1:10
    x=sprand(m, 1,0.025);
    y=sprand(n,1,0.025);
    A_exact=A_exact+(2/i)*x*y.';    
end 
for i=11:100
    x=sprand(m, 1,0.025);
    y=sprand(n,1,0.025);
    A_exact=A_exact+(1/i)*x*y.';
end 


nerr=zeros(5,k);
nerr2=zeros(5,k);
neta_s=zeros(5,k);
neta_p=zeros(5,k);

nTz_hat=zeros(5,k);
nTw_hat=zeros(5,k);


for j=1:5
    Correlated_noise= B*randn(m,n)*G ;
    E=0.1*(norm(A_exact)/norm(Correlated_noise))*Correlated_noise;
    A=A_exact+E;
    [Z,W,U1,V1,SA,SB,SG] = rsvd(A,B,G);
    [Qz,Tz]=qr(Z);
    [Qw,Tw]=qr(W);
    
    sA=diag(SA);
    ee(j)=norm(A);
    for i=1:k
        icol1 = deim(W(:,1:i),i);
        irow1 = deim(Z(:,1:i),i);
        M1=A(:,icol1(1:i))\A/A(irow1(1:i),:);
        RCUR=A(:,icol1(1:i))*M1*A(irow1(1:i),:);
        err(i)=norm(A-RCUR);
        eta_s(i)=(1/smin(Qz(irow1,1:i)));
        eta_p(i)=(1/smin(Qw(icol1,1:i)));

       Tz_hat(i)=norm(Tz(:,i+1:m));
       Tw_hat(i)=norm(Tw(:,i+1:n));
       err2(i)=(eta_s(i)+eta_p(i))*Tz_hat(i)*Tw_hat(i)*sA(i+1);
       
    
    end
    nerr(j,:)=err;
    neta_s(j,:)=eta_s;
    neta_p(j,:)=eta_p;
    nTz_hat(j,:)=Tz_hat;
    nTw_hat(j,:)=Tw_hat;
    nerr2(j,:)=err2;
    
    
end








semilogy(1:50,mean(neta_s),'-r');
hold on;
semilogy(1:50,mean(neta_p),'--r');
semilogy(1:50,mean(nTz_hat),'-g'); 
semilogy(1:50,mean(nTw_hat),'--g'); 
semilogy(1:50,mean(nerr),'-b');  
semilogy(1:50,mean(nerr2),'--b');  
semilogy(1:50,repmat(mean(ee),1,50),'-k');  


legend('$\eta_s$','$\eta_p$','$\widehat T_Z$','$\widehat T_W$',...
   'RSVD-CUR-error', 'RSVD-CUR-bound', 'Norm-$A_E$','interpreter', 'latex')




