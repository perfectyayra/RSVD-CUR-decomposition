function icol = deim(V,k)

icol=zeros(1,k);
[~, icol(1)] = max(abs(V(:,1)));
for j = 2:k
   V(:,j) = V(:,j) - V(:,1:j-1) * (V(icol(1:j-1),1:j-1) \ V(icol(1:j-1),j+1));
   [~, icol(j)] = max(abs(V(:,j)));
end
