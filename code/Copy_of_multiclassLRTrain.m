function model = multiclassLRTrain(x, y, param)
classLabels = unique(y);
numClass = length(classLabels);
numFeats = size(x,1);
numData = size(x,2);

% Initialize weights randomly (Implement the gradient descent)
model.w = randn(numClass, numFeats)*0.01;
%model.w = zeros(numClass, numFeats);
model.classLabels = classLabels;
% % model.w
% % for i = 1:param.maxiter
% %     grad = 0.0;
% %     for k = 1:length(model.classLabels)
% %         grad = grad + transpose(y*transpose(x - (x*transpose(exp(model.w(k,:)*x)))/transpose(sum(exp(model.w*x),1))));
% %     end
% %     grad = bsxfun(@plus, 2*param.lambda*transpose(model.w), grad);
% %     model.w = model.w - transpose(param.eta*grad);
% %     %model.w
% % end
% % model.w
% % % sumexp = 0.0;
% % % for k=1:numClass
% % %     sumexp = sumexp + exp(model.w(k,:)*x);
% % % end

model.w
% % for iter=1:param.maxiter
% % %     sumexp = 0.0;
% % %     for j=1:numClass
% % %         sumexp = sumexp + exp(model.w(j,:)*x);
% % %     end 
% %     sumexp = sum(exp(model.w*x));
% %     for k=1:numClass
% %         grad = 0.0;
% %         for n=1:numData
% %            sumexp = sum(exp(model.w*x(:,n)));
% %            grad = grad - sum((stable_softmax(model.w(k,:)*x(:,n), sumexp) - y(:,n))*x(:,n)) + 2*param.lambda*sum(model.w(k,:));
% %            %grad = grad/numData;
% %            %grad
% %         end
% %         grad = grad/numData
% %         model.w(k,:) = model.w(k,:) - transpose(param.eta*grad);
% %         model.w(k,:);
% %     end
% %     %grad
% %     %model.w = model.w - param.eta*grad;
% % end
% % model.w
% % model.w;
% % 
% % function exps = stable_softmax(X, sumexp)
% % %temp = exp(X(1,:));
% % %exps = temp./sumexp;
% % exps = X/sumexp;

for iter=1:param.maxiter
    for k=1:numClass
        grad = -sum(x*transpose(y - exp(model.w(k,:)*x)/sum(exp(model.w*x)))) + 2*param.lambda*(model.w(k,:));
        grad = grad/numData;
        model.w(k,:) = model.w(k,:) - (param.eta*grad);
        %model.w(k,:)
    end
    %grad
    model.w
    model.w;
end

    %for iter=1:param.maxiter
% % %         model.w = fminunc(@objective, model.w, optimoptions('fminunc','Display','iter'));
% % %     %end
% % %     function obj = objective(w)
% % %         
% % %         classLabels = unique(y);
% % %         numClass = length(classLabels);
% % %         numFeats = size(x,1);
% % %         numData = size(x,2);
% % % 
% % %         %w = randn(numClass, numFeats)*0.01;
% % %         model.classLabels = classLabels;
% % % 
% % %         sum_n = 0.0;
% % %         for n=1:numData
% % %             sum_k = 0.0;
% % %             for k=1:numClass
% % %                 sum_k = sum_k + y(1,n)*(w(k,:)*x(:,n) - logsumexp(w*x(:,n),1));
% % %             end
% % %             sum_n = sum_n - sum_k;
% % %         end
% % %         obj = sum_n + param.lambda*(norm(w)^2);
% % %     end
% % % 
% % % model.w;
% % % end
% % % 
% % % 
% % % function obj = objective(datastruct)
% % % x = datastruct.x;
% % % y = datastruct.y;
% % % param = datastruct.param;
% % % classLabels = unique(y);
% % % numClass = length(classLabels);
% % % numFeats = size(x,1);
% % % numData = size(x,2);
% % % 
% % % w = randn(numClass, numFeats)*0.01;
% % % model.classLabels = classLabels;
% % % 
% % % sum_n = 0.0;
% % % for n=1:numData
% % %     sum_k = 0.0;
% % %     for k=1:numClass
% % %         sum_k = sum_k + y(1,n)*(w(k,:)*x(:,n) - logsumexp(w*x(:,n),1));
% % %     end
% % %     sum_n = sum_n - sum_k;
% % % end
% % % obj = sum_n + param.lambda*(norm(w)^2);
% % % end
% % % 
% % % function s = logsumexp(a, dim)
% % % % Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
% % % % Default is dim = 1 (columns).
% % % % logsumexp(a, 2) will sum across rows instead of columns.
% % % % Unlike matlab's "sum", it will not switch the summing direction
% % % % if you provide a row vector.
% % % 
% % % % Written by Tom Minka
% % % % (c) Microsoft Corporation. All rights reserved.
% % % 
% % % if nargin < 2
% % %   dim = 1;
% % % end
% % % 
% % % % subtract the largest in each column
% % % [y, i] = max(a,[],dim);
% % % dims = ones(1,ndims(a));
% % % dims(dim) = size(a,dim);
% % % a = a - repmat(y, dims);
% % % s = y + log(sum(exp(a),dim));
% % % i = find(~isfinite(y));
% % % if ~isempty(i)
% % %   s(i) = y(i);
% % % end
% % % end
