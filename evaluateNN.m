%%
%%Code for evaluating Exemplar-based Kernel Preserving Embedding on the
%%Approximate Nearest Neighbor.
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   EBEK: Exemplar-based Kernel Preserving Embedding. Ahmed Elbagoury, Rania Ibrahim, Mohamed S. Kamel and Fakhri Karray
%Inputs:
%     data_name: Matlab input file that contains
%            -Matrix fea: n*d matrix, n samples in d-dimensional space
%            -n*1Vector gnd containing the class of each sample
%     num_queries: number of nearest neighbor queroes. 
%     num_neighbors: number of nearest neighbor queroes. 
%     m: The number of dimensions to project the data
%Outputs:
%   recall_avg: a vector containng the average recall of the approximate nearest
%   neighbors  over 10 runs
%   recall_std: a vector containng the standard deviation of the recall of the approximate nearest
%   neighbors  over 10 runs
%   recall_CI: a vector containng the 95% confidence interval of the  recall of the approximate nearest
%   neighbors  over 10 runs
%   precision_avg: a vector containng the average precision of the approximate nearest
%   neighbors  over 10 runs
%   precision_std: a vector containng the standard deviation of the precision of the approximate nearest
%   neighbors  over 10 runs
%   precision_CI: a vector containng the 95% confidence interval of the  precision of the approximate nearest
%   neighbors  over 10 runs
%   run_time_avg: a vector containng the average run_time of the approximate nearest
%   neighbors  over 10 runs
%   run_time_std: a vector containng the standard deviation of the run_time of the approximate nearest
%   neighbors  over 10 runs
%   run_time_CI: a vector containng the 95% confidence interval of the  run_time of the approximate nearest
%   neighbors  over 10 runs
%%

function [recall_avg, precision_avg, run_time_avg, precision_std, recall_std, run_time_std, recall_CI, precision_CI, run_time_CI] = evaluateNN(data_name, num_queries , num_neighbors, m)
    load(data_name);
    %load ('TDT2p2');
    %num_queries = 100;
    %num_neighbors = 10;
    %m = 100;
    A = fea';
    n = size(A, 2);

    num_repeat = 10;
    recall = zeros(10, num_repeat);
    precision = zeros(10, num_repeat);
    run_time = zeros(10, num_repeat);
    run_t2 = zeros(10, num_repeat);


    rng(1);
    for l = 1 : num_repeat
        fprintf('l = %d\n', l);
        queries = randi([1 n],1,num_queries);

        tic;
         %Original data
        S = A(:, queries)' * A;
        for i = 1: num_queries
           S(i, queries(i)) = 0;
        end
        %A_tilde = A(:, queries);
        %S = pdist2(A_tilde', A');
        NN_original = ANN(S, n, num_queries, num_neighbors);
        orig_time = toc;
        flag = 0;
        R = 10;
        ind = 1;

        %Porject the data
        tic;
        [A_tilde, ~] = SimilarityPreserving_CSS_ANN(A, m);
        S_lower = A_tilde(:, queries)' * A_tilde;
        t1 = toc;
        for q = 1: num_queries
            S_lower(q, queries(q)) = 0;
        end
        prev_NN_lower = 0;
        prev_recall = 0;
        while flag == 0
                %tic;
                fprintf('\tR = %d\n', R);
                NN_lower = ANN(S_lower, n, num_queries, R);
                %t2 = toc;
                t2 = 0;
                run_t2(ind, 1) = t2;
                cur_precision = 0;
                cur_recall = 0;
                for i = 1: num_queries
                    ground_truth = NN_original(i, :);
                    predicted = NN_lower(i, :);
                    count = 0;
                    for j = 1 : R
                        for k = 1: num_neighbors
                            if predicted (j) == ground_truth(k)
                               count = count + 1; 
                               break;
                            end
                        end
                    end
                    cur_precision = cur_precision + (count/ R);
                    cur_recall = cur_recall + (count/num_neighbors);
                end
                cur_precision = cur_precision / num_queries;
                cur_recall = cur_recall / num_queries;
                run_time(ind,l) = t1+ t2;
                recall(ind, l) = cur_recall;
                precision(ind, l) = cur_precision;
                ind = ind + 1;
                %fprintf('%d \t %f \t %f\n', R, cur_recall, cur_precision);
                %fprintf('%d\t%f\t%f\t%f\n', R, cur_recall, cur_precision, t1+t2);
                if(cur_recall == 1 || R >= 2010)
                %if(R>=310)
                    flag = 1;
                else
                    R = R +  100;
                end
                prev_NN_lower = NN_lower;
                prev_recall = cur_recall;
        end    
    end

    recall_avg = mean(recall, 2);
    precision_avg = mean(precision, 2);
    run_time_avg = mean(run_time, 2);

    precision_std = std(precision');
    recall_std = std(recall');
    run_time_std = std(run_time');


    recall_CI = recall_std * 1.96 / sqrt(num_repeat);
    recall_CI = recall_CI';
    precision_CI = precision_std * 1.96 / sqrt(num_repeat);
    precision_CI = precision_CI';
    run_time_CI = run_time_std * 1.96 / sqrt(num_repeat);
    run_time_CI = run_time_CI';
end
 


