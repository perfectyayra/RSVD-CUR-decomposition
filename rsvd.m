function [Z,W,U,V,SA,SB,SC,flag] = rsvd(A,B,C)

%RSVD  Restricted singular value decomposition
% function [X,Y,U,V,SA,SB,SC,flag] = rsvd(A,B,C)


% This code is an implementation of the  so-called {\em regular} matrix triplet $(A, B, C)$,
% i.e., $B$ is of full row rank and $C$ is of full column rank


% In:  A: m x n,  B: m x l,  C: d x n,  m >= n,  m <= l,  n <= d
% Out: A = Z SA W',  B = Z SB U',  C = V SC W'
%
% See also GSVD
%
% Reference: Zha 1992  (Zhang, Wei, Chu 2021 contains typos)
% 
% Revision date: June 6, 2023
% (C) Michiel Hochstenbach, Perfect Gidisu 2023

[m,n] = size(A); [m1,l] = size(B); [d,n1] = size(C); flag = 0;
if  m < n, error('Use transpose of A'); end
if (m ~= m1 || n ~= n1), error('Incompatible dimensions for RSVD'); end


[U1,V1,X1,C1,S1] = gsvd(A,C);
if n < d, S1 = S1(1:n,:); end
[U, V2,X2,SB,S2] = gsvd(B'*U1, (C1/S1)'); SB = SB'; S2 = S2';
s = diag(S2); c = s./sqrt(s.^2+1);
if n == d, SC = diag(c);  V = V1*V2; end
if n < d,  V = V1*blkdiag(V2, eye(d-n));  SC = diag0(c,d,n); end
SA = scale(S2, 2, 1./c);
Z = U1*X2;
W = scale(X1*S1*V2, 2, c);

% debugging

if norm(A - Z*SA*W', 'fro') > 1e-8, flag = 1; end
if norm(B - Z*SB*U', 'fro') > 1e-8, flag = 2; end
if norm(C - V*SC*W', 'fro') > 1e-8, flag = 3; end


return



function B = scale(A, n, d)

if nargin < 2 || isempty(n), n = 1; end
if nargin < 3 || isempty(d), if n == 1, d = sum(A,2); else d = sum(A,1); end; end
d(d == 0) = 1;
if n == 1
  if size(d,2) > 1, d = d.'; end; B = d .\ A;
else
  if size(d,1) > 1, d = d.'; end; B = A ./ d;
end



function A = diag0(v, m, n, s)

if nargin < 1 || isempty(v), v = 1:3; end
if nargin < 2 || isempty(m), m = 4; end
if nargin < 3 || isempty(n), n = m; end
if nargin < 4 || isempty(s), s = 0; end  % Sparse?

if ~s, A = diag(v); else, A = spdiag(v); end
k = length(v);
if m > k, A(k+1:m,:) = 0; end
if n > k, A(:,k+1:n) = 0; end
