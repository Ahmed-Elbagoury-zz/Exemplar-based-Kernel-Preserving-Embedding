%%
%%Code for getting m independent columns
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   EBEK: Exemplar-based Kernel Preserving Embedding. Ahmed Elbagoury, Rania Ibrahim, Mohamed S. Kamel and Fakhri Karray
%Inputs:
%    A: d*n matrix that has n samples in d-dimensional space
%    m: The number of required independent columns
%%
function basiccol = get_basiccol(A, m)
    EPSILON =  1e-13;
    basiccol = zeros(m, 1);
    n = size(A, 2);
    ind = 1;
    i = 1;
    is_selected = zeros(n, 1);
    while ind <= m
        if i > n
           break; 
        end
        new_col= A(:, i);
        %cossim = cur_col' * new_col/ (cur_col' * cur_col);
        for j = 1 : ind-1
            cur_col = A(:, basiccol(j));
            dotproduct = cur_col' * new_col/ (cur_col' * cur_col);
            new_col = new_col- dotproduct * cur_col;
        end
        A(:, i) = new_col;
        if(norm(new_col) > EPSILON)
           basiccol(ind) = i;
           ind = ind + 1;
           is_selected(i) = 1;
        end
        i = i +1;
    end
    i = 1;
    while ind <= m
        fprintf('WARNING: Number of columns is greater than the rank!!\n');
        if is_selected(i) == 0
           basiccol(ind) = i;
            ind = ind + 1; 
        end
        i = i + 1;
    end
end
