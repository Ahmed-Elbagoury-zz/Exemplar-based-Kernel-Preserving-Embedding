load('usps');
num_repeat = 10;
A = fea';   %A is in d * n
num_classes = length(unique(gnd));
m_val = [50, 100, 150, 200];
%m_val = [25, 50, 75, 100];
NMI_val = zeros(num_repeat, length(m_val));
rand = zeros(num_repeat, length(m_val));
FMeasure = zeros(num_repeat, length(m_val));
run_time = zeros(num_repeat, length(m_val));
for j = 1: num_repeat
    for i = 1: length(m_val)
       tic;
       m = m_val(i);
       %D = lle(A, size(A, 2) / length(unique(gnd)), m);
       D = lle(A, 20, m);
       cluster_labels = kmeans_(D', 'random', num_classes);
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
