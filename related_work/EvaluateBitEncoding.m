load('PIE68p');
%load ('TDT2p2');
A = fea';
%clear('fea');
%clear('gnd');
d = size(A, 1);
n = size(A, 2);

num_repeat = 10;
m = 100;
num_queries = 100;
%num_training = 100;
num_training = 1000;
method_name = 'ITQ';
num_neighbors = 10;


recall = zeros(10, num_repeat);
precision = zeros(10, num_repeat);
run_time = zeros(10, num_repeat);
run_t2 = zeros(10, num_repeat);


rng(1);
for l = 1: num_repeat
    fprintf('l = %d\n', l);
    
    queries = randi([1 n],1,num_queries);
    %{
     %Original data
    S = A(:, queries)' * A;
    for i = 1: num_queries
       S(i, queries(i)) = 0;
    end
    tic;


    %A_tilde = A(:, queries);
    %S = pdist2(A_tilde', A');
    NN_original = ANN(S, n, num_queries, num_neighbors);
    %}
    name = sprintf('NN_original_TDT2p2_%d_%d', num_neighbors, l);
    %save(name, 'NN_original');
    load(name);
    name = sprintf('queries_TDT2p2_%d_%d', num_neighbors, l);
    %save(name, 'queries');
    load(name);
%    orig_time = toc;
    clear('S');
   
    %load('S_20NGFullp.mat');
    %load('org_NN_20NGFullp_100query_10neighbors');
    
    
    flag = 0;
    R = 10;
    ind = 1;
    %Porject the data
    tic;
    Dham = test(A', m, method_name, num_neighbors, num_training, queries);
    t1 = toc;
    max_dist = max(Dham(:));
    S_lower = max_dist - Dham;
    for q = 1: num_queries
        S_lower(q, queries(q)) = 0;
    end

    prev_NN_lower = 0;
    prev_recall = 0;
    while flag == 0
            fprintf('\tR = %d\n', R);
            %tic;
            NN_lower = ANN(double(S_lower), n, num_queries, R);
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
            run_time(ind, l) = t1+ t2;
            recall(ind, l) = cur_recall;
            precision(ind, l) = cur_precision;
            ind = ind + 1;
            %fprintf('%d\t%f\t%f\t%f\n', R, cur_recall, cur_precision, t1+t2);
            %if(cur_recall == 1 || R >= 2010)
            if(cur_recall == 1 || R >= 160)
                flag = 1;
            else
                R = R+30;
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
