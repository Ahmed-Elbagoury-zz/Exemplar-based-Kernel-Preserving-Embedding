function Dhamm = test(X, bit, method, averageNumberNeighbors, num_training, test_indicies)
%
% demo code for generating small code and evaluation
% input X should be a n*d matrix, n is the number of images, d is dimension
% ''method'' is the method used to generate small code
% ''method'' can be 'ITQ', 'RR', 'LSH' and 'SKLSH' 
%


% parameters

% split up into training and test set
[ndata, D] = size(X);
%R = randperm(ndata);
%Xtest = X(R(1:num_test),:);
%test_indicies = R(1:num_test);
%R(1:num_test) = [];
R = randperm(ndata);
R(test_indicies) = [];
Xtraining = X(R,:);
xtrain = Xtraining(1:num_training, :);
%clear X;

%{
% define ground-truth neighbors (this is only used for the evaluation):
R = randperm(num_training);
DtrueTraining = distMat(Xtraining(R(1:100),:),Xtraining); % sample 100 points to find a threshold
Dball = sort(DtrueTraining,2);
clear DtrueTraining;
Dball = mean(Dball(:,averageNumberNeighbors));
% scale data so that the target distance is 1
Xtraining = Xtraining / Dball;
Xtest = Xtest / Dball;
Dball = 1;
% threshold to define ground truth
DtrueTestTraining = distMat(Xtest,Xtraining);
WtrueTestTraining = DtrueTestTraining < Dball;
clear DtrueTestTraining
%}

% generate training ans test split and the data matrix
XX = X;
%clear('X');
% center the data, VERY IMPORTANT
sampleMean = mean(XX,1);
XX = (XX - repmat(sampleMean,size(XX,1),1));
%{
step = 2000;
for i = 1: step: size(XX,  1)
   if i+ step < size(XX, 1) 
        XX(i: i + step-1, :) = (XX(i: i + step-1, :) - repmat(sampleMean,step,1));
   else
        XX(i:size(XX, 1), :)  = (XX(i:size(XX, 1), :) - repmat(sampleMean,size(XX, 1) -i + 1,1));
   end
end
%}


%several state of art methods
switch(method)
    
    % ITQ method proposed in our CVPR11 paper
    case 'ITQ'
        % PCA
        [pc, l] = eigs(cov(xtrain),bit);
        XX = XX * pc;
        xtrain = xtrain * pc;
        % ITQ
        [Y, R] = ITQ(xtrain,averageNumberNeighbors);
        XX = XX*R;
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
        Y = compactbit(Y>0);
    % RR method proposed in our CVPR11 paper
    case 'RR'
        % PCA
        [pc, l] = eigs(cov(xtrain), bit);
        XX = XX * pc;
        % RR
        R = randn(size(XX,2),bit);
        [U S V] = svd(R);
        XX = XX*U(:,1:bit);
        Y = compactbit(XX>0);
   % SKLSH
   % M. Raginsky, S. Lazebnik. Locality Sensitive Binary Codes from
   % Shift-Invariant Kernels. NIPS 2009.
    case 'SKLSH' 
        RFparam.gamma = 1;
        RFparam.D = D;
        RFparam.M = bit;
        RFparam = RF_train(RFparam);
        B1 = RF_compress(XX, RFparam);
        Y = B1;
    % Locality sensitive hashing (LSH)
     case 'LSH'
        XX = XX * randn(size(XX,2),bit);
        Y = zeros(size(XX));
        Y(XX>=0)=1;
        Y = compactbit(Y);
end

% compute Hamming metric and compute recall precision
B1 = Y;
B2 = Y(test_indicies,:);
Dhamm = hammingDist(B2, B1);
%[recall, precision, rate] = recall_precision(WtrueTestTraining, Dhamm);

% plot the curve
%plot(recall,precision,'-o');
%xlabel('Recall');
%ylabel('Precision');





