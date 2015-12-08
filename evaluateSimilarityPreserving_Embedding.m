%%
%%Code for evaluating Exemplar-based Kernel Preserving Embedding on the
%%clustering. K-Means is used for clustering
%Author: Ahmed Elbagoury (ahmed.elbagoury@uwaterloo.ca)
%If you use this paper, please cite the following paper:
%   EBEK: Exemplar-based Kernel Preserving Embedding. Ahmed Elbagoury, Rania Ibrahim, Mohamed S. Kamel and Fakhri Karray
%Inputs:
%     data_name: Matlab input file that contains
%            -Matrix fea: n*d matrix, n samples in d-dimensional space
%            -n*1Vector gnd containing the class of each sample
%     Kernel_type as specified in kernel.m
%     Kernel info as specified in kernel.m
%     m_val: a vector containing the values of number of dimensions
%Outputs:
%   NMI_avg: a vector containng the average NMI of the output clustering
%   over 10 runs
%   NNI_std: a vector containng the standard deviation of the NMI of the output clustering
%   over 10 runs
%   NMI_CI: a vector containng the 95% confidence interval of the  NMI of the output clustering
%   over 10 runs
%   rand_avg: a vector containng the average rand index of the output clustering
%   over 10 runs
%   rand_std: a vector containng the standard deviation of the rand index of the output clustering
%   over 10 runs
%   rand_CI: a vector containng the 95% confidence interval of the  rand index of the output clustering
%   over 10 runs
%   FMeasure_avg: a vector containng the average FMeasure of the output clustering
%   over 10 runs
%   FMeasure_std: a vector containng the standard deviation of the FMeasure of the output clustering
%   over 10 runs
%   FMeasure_CI: a vector containng the 95% confidence interval of the  FMeasure of the output clustering
%   over 10 runs
%   time_avg: a vector containng the average time of running the clustering
%   over 10 runs
%   time_std: a vector containng the standard deviation of time of running the clustering
%   over 10 runs
%   time_CI: a vector containng the 95% confidence interval of the  time of running the clustering
%   over 10 runs
%   
%%

function [NMI_avg, NMI_std, NMI_CI, rand_avg, rand_std, rand_CI, FMeasure_avg, F_CI, time_avg, time_CI] = evaluateSimilarityPreserving_Embedding(data_name, kernel_type, m_val)
    load(data_name);
    num_repeat = 10;
    A = fea';
    num_clusters = length(unique(gnd));
    NMI_val = zeros(num_repeat, length(m_val));
    rand = zeros(num_repeat, length(m_val));
    FMeasure = zeros(num_repeat, length(m_val));
    run_time = zeros(num_repeat, length(m_val));
    if STRCMP(kernel_type, 'linear') 
        K = A*A;
    else
        K = Kernel2(A', A', KernelInfo);
    end

    %[cluster_labels, ~,~] = kernelkmeans(K, num_clusters);
    for j = 1: num_repeat
        fprintf('j = %d\n', j);
        for i = 1: length(m_val)
            fprintf('\ti = %d\n', i);
            tic;
            m = m_val(i);
            if STRCMP(kernel_type, 'linear') 
                W = SimilarityPreserving_Embedding(A, m);
            else
                W = kernel_SimilarityPreserving_Embedding(K, m);
            end    

            %[U, Sigma, V] = svds(A, m);
            %W = U * Sigma * V';

            %[~, score] = pca(A', 'NumComponents', m);
            %W = score';

            %K_tilde = W'*W;
            cluster_labels = kmeans_(W', 'random', num_clusters);
            run_time(j, i) = toc;
            [rand(j, i), FMeasure(j, i), NMI_val(j, i)] = clusterEvaluator(cluster_labels , gnd );
        end
    end

    NMI_avg = mean(NMI_val);
    NMI_std = std(NMI_val);
    NMI_CI = NMI_std * 1.96 / sqrt(num_repeat);
    rand_avg = mean(rand);
    rand_std = std(rand);
    rand_CI = rand_std * 1.96 / sqrt(num_repeat);
    FMeasure_avg = mean(FMeasure);
    FMeasure_std = std(FMeasure);
    F_CI = FMeasure_std * 1.96 / sqrt(num_repeat);
    time_avg = mean(run_time);
    time_std = std(run_time);
    time_CI =  time_std * 1.96 / sqrt(num_repeat);

    NMI_avg = NMI_avg';
    NMI_CI = NMI_CI';
    FMeasure_avg = FMeasure_avg';
    F_CI = F_CI';
    time_avg = time_avg';
    time_CI = time_CI';
end
