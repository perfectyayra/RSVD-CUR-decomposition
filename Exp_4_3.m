load('mfeat-fou');
load('mfeat-kar');
load('mfeat-pix');
fou=normalize(mfeat_fou);
kar=normalize(mfeat_kar);
pix=normalize(mfeat_pix);



labels=zeros(2000,1);
labels(201:400)=1;labels(401:600)=2;labels(601:800)=3;
labels(801:1000)=4;labels(1001:1200)=5;labels(1201:1400)=6;
labels(1401:1600)=7; labels(1601:1800)=8; labels(1801:2000)=9;


%% first case pix and fou
for i=1:20
    rng(i)
    cv1 = cvpartition(size(labels,1),'HoldOut',0.25);
    idx1 = cv1.test;
    B = pix(~idx1,:);
    B_t  = pix(idx1,:);
    G = fou(~idx1,:);
    G_t  = fou(idx1,:);
    y_train=labels(~idx1,:);
    y_test=labels(idx1,:);


    BG=[B G];
    BG_t=[B_t G_t];

    [~,~,V_B]=svd(B,0);
    [~,~,V_G]=svd(G,0);
    [~,~,V_BG]=svd(BG,0);
    

    [Z,W,~,~,~,~,~] = rsvd(B'*G,B',G);
  
    s=[20,30];
    for j=1:length(s)
        k=s(j);
        kk=k/2;

        %% QDEIM RSVD-ID selection of B and G

        icolB = qdeim(Z(:,1:k),k);
        icolG = qdeim(W(:,1:k),k);

         % train set
        C_B=B(:,icolB);
        C_G=G(:,icolG);


        icolB1 = qdeim(Z(:,1:kk),kk);
        icolG1 = qdeim(W(:,1:kk),kk);

        C_B1=B(:,icolB1);
        C_G1=G(:,icolG1);
        C_B_G=[C_B1 C_G1]; % Fused RSVD-ID of B and G
        
         %test set 
        C_B_t=B_t(:,icolB);
        C_G_t=G_t(:,icolG);
        C_B1_t=B_t(:,icolB1);
        C_G1_t=G_t(:,icolG1);
        C_B_G_t=[C_B1_t C_G1_t];


         % train model using RSVD-ID selection of G and G
    
        cl1 = fitcknn(C_B,y_train);
        cl2 = fitcknn(C_G,y_train);
        cl3 = fitcknn(C_B_G,y_train); % Fused RSVD-ID of B and G

        % predict labels 
 
        label1 = predict(cl1,C_B_t);
        label2 = predict(cl2,C_G_t);

        label3 = predict(cl3,C_B_G_t);

        % classification performance 
        cp1 = classperf(y_test,label1);
        cp2 = classperf(y_test,label2);
        cp3 = classperf(y_test,label3);

        % error rate 
        RSVD_ID_v1(i,j)=cp1.ErrorRate;
        RSVD_ID_v2(i,j)=cp2.ErrorRate;
        FUSED_RSVD_ID(i,j)=cp3.ErrorRate;

        %% QDEIM ID selection of B and G 
        
        icolB2 = qdeim(V_B(:,1:k),k);
        icolG2 = qdeim(V_G(:,1:k),k);
        
        % train set
        C_B2=B(:,icolB2);
        C_G2=G(:,icolG2);

        icolB21 = qdeim(V_B(:,1:kk),kk);
        icolG21 = qdeim(V_G(:,1:kk),kk);
        C_B21=B(:,icolB21);
        C_G21=G(:,icolG21);
        C_B_G2=[C_B21 C_G21]; % Fused ID of B and G

        % test set 
        C_B2_t=B_t(:,icolB2);
        C_G2_t=G_t(:,icolG2);

        C_B21_t=B_t(:,icolB21);
        C_G21_t=G_t(:,icolG21);
        C_B_G2_t=[C_B21_t C_G21_t];

        % train model using ID selection of B and G
        cl4 = fitcknn(C_B2,y_train);
        cl5 = fitcknn(C_G2,y_train);
        cl6 = fitcknn(C_B_G2,y_train); % Fused ID of B and G
        
        % predict
        label4 = predict(cl4,C_B2_t);
        label5 = predict(cl5,C_G2_t); 
        label6 = predict(cl6,C_B_G2_t);

         % classification performance 
        cp4 = classperf(y_test,label4);
        cp5 = classperf(y_test,label5);
        cp6 = classperf(y_test,label6);

        % error rate 
        ID_v1(i,j)=cp4.ErrorRate;
        ID_v2(i,j)=cp5.ErrorRate;
        FUSED_ID(i,j)=cp6.ErrorRate;



        %% QDEIM ID selection of concat BG
        icolBG = qdeim(V_BG(:,1:k),k);

        %train set
        C_BG=BG(:,icolBG);

        %test set

        C_BG_t=BG_t(:,icolBG);
      
        % train model using ID selection of concat BG
        cl7 = fitcknn(C_BG,y_train);
        
        % predict labels 
        label7 = predict(cl7,C_BG_t);
       
         %classification performance 
         cp7 = classperf(y_test,label7);
         % error rate 
         Concat_ID(i,j)=cp7.ErrorRate;
    
    end


end
Data=['B=pix vs. G=fou';'B=pix vs. G=fou'];
rank_k=[20;30];
ID_view1=(mean(ID_v1))';
ID_view2=(mean(ID_v2))';
RSVD_ID_view1=(mean(RSVD_ID_v1))';
RSVD_ID_view2=(mean(RSVD_ID_v2))';



pix_fou_sing_view=table(Data,rank_k,ID_view1,RSVD_ID_view1,ID_view2,RSVD_ID_view2);


Fused_RSVD_ID=(mean(FUSED_RSVD_ID))';
Fused_ID=(mean(FUSED_ID))';
Concat_ID=(mean(Concat_ID))';
pix_fou_fused_view=table(Data,rank_k,Fused_ID,Concat_ID,Fused_RSVD_ID);
 
clearvars -except fou kar pix labels rank_k pix_fou_sing_view pix_fou_fused_view

%% second case fou and kar
for i=1:20
    rng(i)
    cv1 = cvpartition(size(labels,1),'HoldOut',0.25);
    idx1 = cv1.test;
    B = fou(~idx1,:);
    B_t  = fou(idx1,:);
    G = kar(~idx1,:);
    G_t  = kar(idx1,:);
    y_train=labels(~idx1,:);
    y_test=labels(idx1,:);


    BG=[B G];
    BG_t=[B_t G_t];

    [~,~,V_B]=svd(B,0);
    [~,~,V_G]=svd(G,0);
    [~,~,V_BG]=svd(BG,0);
    

    [Z,W,~,~,~,~,~] = rsvd(B'*G,B',G);
  
    s=[20,30];
    for j=1:length(s)
        k=s(j);
        kk=k/2;

        %% QDEIM RSVD-ID selection of B and G

        icolB = qdeim(Z(:,1:k),k);
        icolG = qdeim(W(:,1:k),k);

         % train set
        C_B=B(:,icolB);
        C_G=G(:,icolG);


        icolB1 = qdeim(Z(:,1:kk),kk);
        icolG1 = qdeim(W(:,1:kk),kk);

        C_B1=B(:,icolB1);
        C_G1=G(:,icolG1);
        C_B_G=[C_B1 C_G1]; % Fused RSVD-ID of B and G
        
         %test set 
        C_B_t=B_t(:,icolB);
        C_G_t=G_t(:,icolG);
        C_B1_t=B_t(:,icolB1);
        C_G1_t=G_t(:,icolG1);
        C_B_G_t=[C_B1_t C_G1_t];


         % train model using RSVD-ID selection of G and G
    
        cl1 = fitcknn(C_B,y_train);
        cl2 = fitcknn(C_G,y_train);
        cl3 = fitcknn(C_B_G,y_train); % Fused RSVD-ID of B and G

        % predict labels 
 
        label1 = predict(cl1,C_B_t);
        label2 = predict(cl2,C_G_t);

        label3 = predict(cl3,C_B_G_t);

        % classification performance 
        cp1 = classperf(y_test,label1);
        cp2 = classperf(y_test,label2);
        cp3 = classperf(y_test,label3);

        % error rate 
        RSVD_ID_v1(i,j)=cp1.ErrorRate;
        RSVD_ID_v2(i,j)=cp2.ErrorRate;
        FUSED_RSVD_ID(i,j)=cp3.ErrorRate;

        %% QDEIM ID selection of B and G 
        
        icolB2 = qdeim(V_B(:,1:k),k);
        icolG2 = qdeim(V_G(:,1:k),k);
        
        % train set
        C_B2=B(:,icolB2);
        C_G2=G(:,icolG2);

        icolB21 = qdeim(V_B(:,1:kk),kk);
        icolG21 = qdeim(V_G(:,1:kk),kk);
        C_B21=B(:,icolB21);
        C_G21=G(:,icolG21);
        C_B_G2=[C_B21 C_G21]; % Fused ID of B and G

        % test set 
        C_B2_t=B_t(:,icolB2);
        C_G2_t=G_t(:,icolG2);

        C_B21_t=B_t(:,icolB21);
        C_G21_t=G_t(:,icolG21);
        C_B_G2_t=[C_B21_t C_G21_t];

        % train model using ID selection of B and G
        cl4 = fitcknn(C_B2,y_train);
        cl5 = fitcknn(C_G2,y_train);
        cl6 = fitcknn(C_B_G2,y_train); % Fused ID of B and G
        
        % predict
        label4 = predict(cl4,C_B2_t);
        label5 = predict(cl5,C_G2_t); 
        label6 = predict(cl6,C_B_G2_t);

         % classification performance 
        cp4 = classperf(y_test,label4);
        cp5 = classperf(y_test,label5);
        cp6 = classperf(y_test,label6);

        % error rate 
        ID_v1(i,j)=cp4.ErrorRate;
        ID_v2(i,j)=cp5.ErrorRate;
        FUSED_ID(i,j)=cp6.ErrorRate;



        %% QDEIM ID selection of concat BG
        icolBG = qdeim(V_BG(:,1:k),k);

        %train set
        C_BG=BG(:,icolBG);

        %test set

        C_BG_t=BG_t(:,icolBG);
      
        % train model using ID selection of concat BG
        cl7 = fitcknn(C_BG,y_train);
        
        % predict labels 
        label7 = predict(cl7,C_BG_t);
       
         %classification performance 
         cp7 = classperf(y_test,label7);
         % error rate 
         Concat_ID(i,j)=cp7.ErrorRate;
    
    end


end
Data=['B=fou vs. G=kar';'B=fou vs. G=kar'];
ID_view1=(mean(ID_v1))';
ID_view2=(mean(ID_v2))';
RSVD_ID_view1=(mean(RSVD_ID_v1))';
RSVD_ID_view2=(mean(RSVD_ID_v2))';



fou_kar_sing_view=table(Data,rank_k,ID_view1,RSVD_ID_view1,ID_view2,RSVD_ID_view2);


Fused_RSVD_ID=(mean(FUSED_RSVD_ID))';
Fused_ID=(mean(FUSED_ID))';
Concat_ID=(mean(Concat_ID))';
fou_kar_fused_view=table(Data,rank_k,Fused_ID,Concat_ID,Fused_RSVD_ID);

clearvars -except fou kar pix labels rank_k pix_fou_sing_view pix_fou_fused_view fou_kar_sing_view fou_kar_fused_view

%% third case pix and kar
for i=1:20
    rng(i)
    cv1 = cvpartition(size(labels,1),'HoldOut',0.25);
    idx1 = cv1.test;
    B = pix(~idx1,:);
    B_t  = pix(idx1,:);
    G = kar(~idx1,:);
    G_t  = kar(idx1,:);
    y_train=labels(~idx1,:);
    y_test=labels(idx1,:);


    BG=[B G];
    BG_t=[B_t G_t];

    [~,~,V_B]=svd(B,0);
    [~,~,V_G]=svd(G,0);
    [~,~,V_BG]=svd(BG,0);
    

    [Z,W,~,~,~,~,~] = rsvd(B'*G,B',G);
  
    s=[20,30];
    for j=1:length(s)
        k=s(j);
        kk=k/2;

        %% QDEIM RSVD-ID selection of B and G

        icolB = qdeim(Z(:,1:k),k);
        icolG = qdeim(W(:,1:k),k);

         % train set
        C_B=B(:,icolB);
        C_G=G(:,icolG);


        icolB1 = qdeim(Z(:,1:kk),kk);
        icolG1 = qdeim(W(:,1:kk),kk);

        C_B1=B(:,icolB1);
        C_G1=G(:,icolG1);
        C_B_G=[C_B1 C_G1]; % Fused RSVD-ID of B and G
        
         %test set 
        C_B_t=B_t(:,icolB);
        C_G_t=G_t(:,icolG);
        C_B1_t=B_t(:,icolB1);
        C_G1_t=G_t(:,icolG1);
        C_B_G_t=[C_B1_t C_G1_t];


         % train model using RSVD-ID selection of G and G
    
        cl1 = fitcknn(C_B,y_train);
        cl2 = fitcknn(C_G,y_train);
        cl3 = fitcknn(C_B_G,y_train); % Fused RSVD-ID of B and G

        % predict labels 
 
        label1 = predict(cl1,C_B_t);
        label2 = predict(cl2,C_G_t);

        label3 = predict(cl3,C_B_G_t);

        % classification performance 
        cp1 = classperf(y_test,label1);
        cp2 = classperf(y_test,label2);
        cp3 = classperf(y_test,label3);

        % error rate 
        RSVD_ID_v1(i,j)=cp1.ErrorRate;
        RSVD_ID_v2(i,j)=cp2.ErrorRate;
        FUSED_RSVD_ID(i,j)=cp3.ErrorRate;

        %% QDEIM ID selection of B and G 
        
        icolB2 = qdeim(V_B(:,1:k),k);
        icolG2 = qdeim(V_G(:,1:k),k);
        
        % train set
        C_B2=B(:,icolB2);
        C_G2=G(:,icolG2);

        icolB21 = qdeim(V_B(:,1:kk),kk);
        icolG21 = qdeim(V_G(:,1:kk),kk);
        C_B21=B(:,icolB21);
        C_G21=G(:,icolG21);
        C_B_G2=[C_B21 C_G21]; % Fused ID of B and G

        % test set 
        C_B2_t=B_t(:,icolB2);
        C_G2_t=G_t(:,icolG2);

        C_B21_t=B_t(:,icolB21);
        C_G21_t=G_t(:,icolG21);
        C_B_G2_t=[C_B21_t C_G21_t];

        % train model using ID selection of B and G
        cl4 = fitcknn(C_B2,y_train);
        cl5 = fitcknn(C_G2,y_train);
        cl6 = fitcknn(C_B_G2,y_train); % Fused ID of B and G
        
        % predict
        label4 = predict(cl4,C_B2_t);
        label5 = predict(cl5,C_G2_t); 
        label6 = predict(cl6,C_B_G2_t);

         % classification performance 
        cp4 = classperf(y_test,label4);
        cp5 = classperf(y_test,label5);
        cp6 = classperf(y_test,label6);

        % error rate 
        ID_v1(i,j)=cp4.ErrorRate;
        ID_v2(i,j)=cp5.ErrorRate;
        FUSED_ID(i,j)=cp6.ErrorRate;



        %% QDEIM ID selection of concat BG
        icolBG = qdeim(V_BG(:,1:k),k);

        %train set
        C_BG=BG(:,icolBG);

        %test set

        C_BG_t=BG_t(:,icolBG);
      
        % train model using ID selection of concat BG
        cl7 = fitcknn(C_BG,y_train);
        
        % predict labels 
        label7 = predict(cl7,C_BG_t);
       
         %classification performance 
         cp7 = classperf(y_test,label7);
         % error rate 
         Concat_ID(i,j)=cp7.ErrorRate;
    
    end


end
Data=['B=pix vs. G=kar';'B=pix vs. G=kar'];
ID_view1=(mean(ID_v1))';
ID_view2=(mean(ID_v2))';
RSVD_ID_view1=(mean(RSVD_ID_v1))';
RSVD_ID_view2=(mean(RSVD_ID_v2))';



pix_kar_sing_view=table(Data,rank_k,ID_view1,RSVD_ID_view1,ID_view2,RSVD_ID_view2);


Fused_RSVD_ID=(mean(FUSED_RSVD_ID))';
Fused_ID=(mean(FUSED_ID))';
Concat_ID=(mean(Concat_ID))';
pix_kar_fused_view=table(Data,rank_k,Fused_ID,Concat_ID,Fused_RSVD_ID);
 
 
Table3=outerjoin(pix_fou_sing_view,fou_kar_sing_view,'MergeKeys',true);
Table3=outerjoin(Table3,pix_kar_sing_view,'MergeKeys',true)

Table4=outerjoin(pix_fou_fused_view,fou_kar_fused_view,'MergeKeys',true);
Table4=outerjoin(Table4,pix_kar_fused_view,'MergeKeys',true)
 
