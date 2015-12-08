function [K, KernelInfo] = Kernel(X, Y, KernelInfo, XL, YL)
if strcmp(KernelInfo.Type, 'Linear') 
    K = X*Y';            
    
elseif strcmp(KernelInfo.Type, 'Polynomial')
    if ~isfield(KernelInfo, 'Coeff')
        KernelInfo.Coeff = 1;
    end
    if ~isfield(KernelInfo, 'Power')
        KernelInfo.Power = 5;
    end
    
    K = (X*Y'+ KernelInfo.Coeff).^KernelInfo.Power;  

elseif strcmp(KernelInfo.Type, 'Neural')
    if ~isfield(KernelInfo, 'a')
        KernelInfo.a = 0.0045;
    end
    if ~isfield(KernelInfo, 'b')
        KernelInfo.b = 0.11;
    end
    
    K = tanh(KernelInfo.a*X*Y'+ KernelInfo.b);
    
elseif strcmp(KernelInfo.Type, 'RBF')     
    if ~isfield(KernelInfo, 'Sigma')
        KernelInfo.Sigma = 0;
    end
    
    K =  X*Y';                  % Inner products    
   
    if ~exist('XL', 'var')
       XL = sum(X.^2, 2);           % The lenghts of X vectors squared
    end
    
    if ~exist('YL', 'var')
       YL = sum(Y.^2, 2);           % The lenghts of Y vectors squared
    end
    
    eX = ones(size(X, 1), 1);             
    eY = ones(size(Y, 1), 1);
    D2 = eX*YL' + XL*eY' - 2*K;  % This is more efficient than squareform(pdist(X)).^2   
    D2 = D2 .* (D2>0);
    
   if KernelInfo.Sigma == 0
        Sigma2 = mean(D2(:));       
        K = exp(-D2/(2*Sigma2));
        KernelInfo.Sigma = sqrt(Sigma2);
    else
        K = exp(-D2/(2.*KernelInfo.Sigma.^2));
    end
end

% % Construct a kNN kernerl
% if isfield(KernelInfo, 'kNN')
%     if KernelInfo.kNN > 0       
%         [n k] = size(K);        
%         r = KernelInfo.kNN;
%         dump = zeros(n, r);
%         idx = dump;
%         for i = 1:r
%             [dump(:,i),idx(:,i)] = max(K,[],2);
%             temp = (idx(:,i)-1)*n+[1:n]';
%             K(temp) = -1e100;
%         end
%         % dump = exp(-dump/(2*sigma^2));
%         % sumD = sum(dump,2);
%         Gsdx = dump; % bsxfun(@rdivide,dump,sumD);
%         Gsdx
%         Gidx = repmat([1:n]',1,r);
%         Gjdx = idx;
%         K =sparse(Gidx(:),Gjdx(:),Gsdx(:), n, k);
%     end
% end

end

