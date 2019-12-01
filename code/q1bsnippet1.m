function features = pixelFeatures(x)
features = reshape(x, [784, size(x, 4)]); 
%features = sqrt(features);
%features = features/norm(features); 
features = features - mean(features)/std(features);