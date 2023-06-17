% this code produces the results in the first row of table 7
% uncomment the other problems to get the results of the problems(other
% rows)
m = 1000; n = 100; l = m; d = n; 
s=[10,20,30];
for j=1:100    
    y = randn(m,1); z=randn(d,1);
    b = [y; z]; 

    %problem 1

   A = gallery('randsvd',[m n],10); 
   B = gallery('randsvd',[m l],10^4); 
   G = randn(d,n);

    %problem 2
%    A = gallery('randsvd',[m n],10^6); 
%    B = gallery('randsvd',[m l],10); 
%    G = randn(d,n);


    %problem 3

%    A = gallery('randsvd',[m n],10^4); 
%    B = gallery('randsvd',[m l],10^4); 
%    G = gallery('randsvd',[d n],10);


    %problem 4

%     A = randn(m,n); 
%     B = randn(m,l); 
%     G = randn(d,n);


    for i=1:length(s)
        k=s(i);         
        [Z,W,U,V,SA,SB,SC] = rsvd(A,B,G);
        
        p = deim(W(:,1:k),k);       
       
        E1 = [A(:,p) B; G(:,p) zeros(d,l)];       
        
        x1 = E1 \ b;   
        e1(j,i)=norm(E1*x1-b)/norm(b);          
       
    end
end

mean(e1)

    
    
