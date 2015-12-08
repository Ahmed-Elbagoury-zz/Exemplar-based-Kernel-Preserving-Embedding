%Code for Approximate Nearest Neighbor 
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   EBEK: Exemplar-based Kernel Preserving Embedding. Ahmed Elbagoury, Rania Ibrahim, Mohamed S. Kamel and Fakhri Karray
%Inputs:
    %n: number of samples
    %S: A n*n matrix of similarity between the samples
    %num_queries: number of nearest neighbor queroes. 
    %num_neighbors: number of nearest neighbor queroes. 
function NN = ANN(S, n, num_queries, num_neighbors)
NN = zeros(num_queries, num_neighbors);
loop_count = 0;
if_count = 0;
for k = 1 : num_queries
    indexes = 1:1:n;
    for i = 1: num_neighbors %Get the smallest num_queries numbers
        for j = i + 1 : n
            loop_count = loop_count + 1;
            if S(k, j) >  S(k, i)
                if_count = if_count + 1;
                %Swap the values and the indexes
                temp = S(k, i);
                S(k, i) = S(k, j);
                S(k, j) = temp;
                temp = indexes(i);
                indexes(i) = indexes(j);
                indexes(j) = temp;
            end
        end
    end
    for i = 1 : num_neighbors
       NN(k, i) = indexes(i); 
    end
end
