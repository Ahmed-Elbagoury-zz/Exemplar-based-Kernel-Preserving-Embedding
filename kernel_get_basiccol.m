%%
%%Code for getting m independent columns for arbitrary kernel matrices
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   EBEK: Exemplar-based Kernel Preserving Embedding. Ahmed Elbagoury, Rania Ibrahim, Mohamed S. Kamel and Fakhri Karray
%Inputs:
%    A: d*n matrix that has n samples in d-dimensional space
%    m: The number of required independent columns
%%
function basiccol = kernel_get_basiccol(K, m)
    EPSILON =  1e-13;
    basiccol = zeros(m, 1);
    basic_ind = 1;
    n = size(K, 2);
    is_deleted = zeros(n, 1);
    i = 1;
    
    while basic_ind <= m
        while is_deleted(i) == 1
            i = i + 1;
        end
        cur_col = K(is_deleted == 0, i);
        n = size(K, 2);
        basiccol(basic_ind) = i;
        basic_ind = basic_ind + 1;
        for j = i+1 : n
           if is_deleted(j) == 1
               continue;
           end
           new_col  = K(is_deleted == 0, j);
           dotproduct = cur_col' * new_col/ (cur_col' * cur_col);
           new_col = new_col- dotproduct * cur_col; 
           K(is_deleted == 0, j) = new_col;
           if(norm(new_col) < EPSILON)
               is_deleted(j) = 1;
               cur_col(j) = [];
            end
        end
        i = i + 1;
    end
end
