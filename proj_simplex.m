function [output] = proj_simplex(x_input)
    [m,n] = size(x_input);
    X = sort(x_input, 1,'descend');
    Xtmp =diag(sparse(1./(1:m))) * (cumsum(X,1)-1);
    output=max(bsxfun(@minus,x_input,Xtmp(sub2ind([m,n],sum(x_input>Xtmp,1),1:n))),0);
end