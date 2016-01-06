%%
%This code generates 1000 matrix and computes the low rank approximation
%of the matrox S = A' * A using truncated SVD, then it computes the low rank approximation of S
%based on subset of columns of A (S2) such that this low rank approximation
%is equal to the low rank approximation obtained by truncated SVD
%%
function [A_tilde, basiccol] = SimilarityPreserving_CSS_ANN(A, m)
    tic;
    [~, Sigma, V] = StochasticSVD(A, m);
    basiccol = get_basiccol(A, m);
    A1 = A(:, basiccol(1:m));
    S_tilde = A1' * A1;
    p = reduceMatrix(S_tilde);
    T = p *  Sigma * V';
    
    A_tilde = A1*T;
end



