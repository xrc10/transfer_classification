function [ XNew ] = extSparseDim( X, dimType, p )
%CHGSPARSEDIM Summary of this function goes here
%   if dimType == 1, change first dimension
%   if dimType == 2, change second dimension

[t1,t2,t3] = find(X);

if dimType == 1
    XNew = sparse(t1,t2,t3,p,size(X,2));
elseif dimType == 2
    XNew = sparse(t1,t2,t3,size(X,1),p);
end



end

