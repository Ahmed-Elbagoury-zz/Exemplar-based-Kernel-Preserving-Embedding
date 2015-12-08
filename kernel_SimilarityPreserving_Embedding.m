%%
%%Code for projecting a data matrix to a space that preserve the arbitray kernel matrix
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   EBEK: Exemplar-based Kernel Preserving Embedding. Ahmed Elbagoury, Rania Ibrahim, Mohamed S. Kamel and Fakhri Karray
%Inputs:
%    A: n*n matrix that has n samples in d-dimensional space
%    m: The number of dimensions that the data will be projected to
%%
function [T, basiccol] = kernel_SimilarityPreserving_Embedding(K, m)
    tic;
    [~, Sigma, V] = StochasticSVD(K, m);
    basiccol = kernel_get_basiccol(K, m);
    S_tilde = K(basiccol, basiccol);
    p = reduceMatrix(S_tilde);
    T = p *  Sigma .^ 0.5 * V';
end
