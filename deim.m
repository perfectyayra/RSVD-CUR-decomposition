function icol = deim(V,k)

icol=zeros(1,k);
for j = 1:k
  [~, icol(j)] = max(abs(V(:,j)));
  if j<k
   V(:,j+1) = V(:,j+1) - V(:,1:j) * (V(icol(1:j),1:j) \ V(icol(1:j),j+1));
  end
end
