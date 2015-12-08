
%%
%%Code for projecting a data matrix to a space that preserve the kernel matrix
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   EBEK: Exemplar-based Kernel Preserving Embedding. Ahmed Elbagoury, Rania Ibrahim, Mohamed S. Kamel and Fakhri Karray
%Inputs:
%    A: d*n matrix that has n samples in d-dimensional space
%    m: The number of dimensions that the data will be projected to
%%
function [W, basiccol] = SimilarityPreserving_Embedding(A, m)
    tic;
    [~, Sigma, V] = StochasticSVD(A, m);
    basiccol = get_basiccol(A, m);
    A1 = A(:, basiccol(1:m));
    S_tilde = A1' * A1;
    p = reduceMatrix(S_tilde);
    T = p *  Sigma * V';
    [~, R] = qr(A1, 0);
    W = R * T;
end



  
