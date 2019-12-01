function features = lbpFeatures(x)
    % Pre-made binary pattern for binary --> decimal mapping
    bin = [1,2,4;8,0,16;32,64,128];
    
    % Store features for all images here
    features = [];
    
    % For each image
    for image = 1:size(x, 4)
        
        % Create a histogram of integer representation of each 3x3 patch
        hist = zeros(256,1);
        im = x(:,:,1,image);
        
        % Make sure 3x3 patches are within boundary limits
        for i = 1:size(im, 1)-2
            for j = 1:size(im,2)-2
                % Get a patch
                patch = im(i:i+3-1, j:j+3-1);
                
                % Subtract from each pixel, the value of centre pixel
                patch = patch - patch(2,2);
                % Assign 0, if the pixel is less than 0
                patch(patch < 0) = 0;
                % Assign 1, if the pixel is greater than 0
                patch(patch > 0) = 1;
                % Multiply the threshold with bin to get an integer representation
                patch = patch .* bin;
                % Get the integer value between 1-256 and put it in histogram
                hist(sum(sum(patch, 1),2)+1) = hist(sum(sum(patch, 1),2)+1) + 1;               
            end
        end
   
        % Taking square root
        %hist = sqrt(hist);
        % l2 normalization
        hist = hist/norm(hist);
        
        % Concatenate histograms for all images
        features = [features hist];
    end