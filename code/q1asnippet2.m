classLabels = unique(y);
x = [x; ones(1, size(x,1))]; % Bias terms absorbed for each feature vector
numClass = length(classLabels);
numFeats = size(x,1);   % There would be one extra feature now, because of the absorbed bias
numData = size(x,2);

% Initialize weights randomly (Implement the gradient descent)
model.w = randn(numClass, numFeats)*0.01;
model.classLabels = classLabels;