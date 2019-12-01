function model = multiclassLRTrain(x, y, param)
classLabels = unique(y);
x = [x; ones(size(x,2), 1)];
numClass = length(classLabels);
numFeats = size(x,1);
numData = size(x,2);

% Initialize weights randomly (Implement the gradient descent)
model.w = randn(numClass, numFeats)*0.01;
model.classLabels = classLabels;

% Iterate for pre-specified maximum iteration count 
for iter=1:param.maxiter
    % For each class
    for k=1:numClass
        % Calculate the gradient of the objective function
        grad = -sum(x*transpose(y - exp(model.w(k,:)*x)/sum(exp(model.w*x)))) + 2*param.lambda*(model.w(k,:));
        grad = grad/numData;
        % Update the parameters
        model.w(k,:) = model.w(k,:) - (param.eta*grad);
        %model.w(k,:)
    end
    %grad
    %model.w
end