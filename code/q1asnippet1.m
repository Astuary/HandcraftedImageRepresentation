function sm = softmax(model,x)

% Multiply w (numClasses, numFeatures) with x (numFeatures, numDataTuples)
term = model.w * x;
% Take exponential of all the elements in term (numClasses, numDataTuples) matrix
exps = exp(term);
% Devide exponents by sum of all exponents for that specific tuple for all classes; that's why dimension=1 in sum() function
sm = exps / sum(exps, 1);
% Additionally, we know that we only assign the class index of the highest probability element from sm(:, numDataTuples) to assign a class label to that specific tuple

% Take the index of the maximum element from each row (dimension = 1)/ data tuple of probability result matrix
[~, ypred] = max(sm, [], 1);
