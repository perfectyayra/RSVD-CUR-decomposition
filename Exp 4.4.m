% This code reproduces the results for noise level epsilon=0.001, change this value to 0.0001 the other result
epsilon=0.001;
for j=1:10
    traindata=load("ann-train.data");
    testdata=load("ann-test.data");
    % normalize data
    A_train=normalize(traindata(:,1:20));
    A_test=normalize(testdata(:,1:20));

    % generate noise
    [m,n]=size(A_train);
    B=chol(toeplitz(0.99.^(0:m-1)));
    E=B*randn(m,n);

    %perturb train data with noise
    A_E=A_train+epsilon*(norm(A_train)/norm(E))*E;

    %cost of variables matrix
    G=diag([ones(16,1);22.78;11.41;14.51;11.41]);
    g=[ones(16,1);22.78;11.41;14.51;11.41]; %cost vector


    % labels 
    y_train=traindata(:,22);
    
    y_test=testdata(:,22);
    
    
    
   %run svd, gsvd, rsvd
    [U,S,V]=svd(A_E,0);
    [~,~,X2,~,~]=gsvd(A_E,G,0);
    X2=fliplr(X2);
    [~,W,~,~,~,~,~] = rsvd(A_E,B',G);


    k=10;

    p = deim(W(:,1:k),k); % RSVD_ID slected column indices
    RSVD_ID_Ca=A_E(:,p);


    p1 = deim(V(:,1:k),k); % ID selected column indices
    ID_Ca=A_E(:,p1);
    
    p2 = deim(X2(:,1:k),k); %GCUR selected column indices
    GCUR_Ca=A_E(:,p2);
    
   % train models 
    cl = fitcknn(A_E,y_train);
    cl1 = fitcknn(RSVD_ID_Ca,y_train);
    cl2 = fitcknn(ID_Ca,y_train);
    cl3= fitcknn(GCUR_Ca,y_train);
    
    RSVD_ID_Ca_test=A_test(:,p);
    ID_Ca_test=A_test(:,p1);
    GCUR_Ca_t=A_test(:,p2);
    
    label = predict(cl,A_test);
    label1 = predict(cl1,RSVD_ID_Ca_test);
    label2 = predict(cl2,ID_Ca_test);
    label3 = predict(cl3,GCUR_Ca_t);
    
    cp = classperf(y_test,label);
    cp1 = classperf(y_test,label1);
    cp2 = classperf(y_test,label2);
    cp3 = classperf(y_test,label3);
    
    AllVar_acc(j)=cp.ErrorRate;
    AllVar_cost(j)=sum(g);
    RSVD_ID_Var_acc(j)=cp1.ErrorRate;
    RSVD_ID_Var_cost(j)=sum(g(p));
    ID_Var_acc(j)=cp2.ErrorRate;
    ID_Var_cost(j)=sum(g(p1));
    GCUR_Var_acc(j)=cp3.ErrorRate;
    GCUR_Var_cost(j)=sum(g(p2));

end
% Error rate 
mean(AllVar_acc)
mean(RSVD_ID_Var_acc)
mean(ID_Var_acc)
mean(GCUR_Var_acc)



% Cost of variables 
mean(AllVar_cost)
mean(RSVD_ID_Var_cost)
mean(ID_Var_cost)
mean(GCUR_Var_cost)

