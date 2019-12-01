function ypred = multiclassLRPredict(model,x)

x = [x; ones(size(x,2), 1)];
numData = size(x,2);
% Simply predict the first class (Implement this)
% ypred = model.classLabels(1)*ones(1, numData);

a = sum(exp(model.w*x),1);
temp = exp(model.w*x)./a;
[~, ypred] = max(temp);
