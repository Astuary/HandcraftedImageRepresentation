function features = extractDigitFeatures(x, featureType)
% EXTRACTDIGITFEATURES extracts features from digit images
%   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
%   images X of the provided FEATURETYPE. The images are assumed to the of
%   size [W H 1 N] where the first two dimensions are the width and height.
%   The output is of size [D N] where D is the size of each feature and N
%   is the number of images. 

switch featureType
    case 'pixel'
        features = pixelFeatures(x);  % implement this
    case 'hog'
        features = hogFeatures(x);  % implement this
    case 'lbp'
        features = lbpFeatures(x);  % implement this
end

function features = zeroFeatures(x)
features = zeros(10, size(x,4));

function features = pixelFeatures(x)
features = reshape(x, [784, size(x, 4)]); 
%features = sqrt(features);
%features = features/norm(features); 
features = features - mean(features)/std(features);
  
function features = hogFeatures(x)
    gamma = 3.1;
    binSize = 4;
    numOri = 8;
    % Store features of all images here
    features = [];
    
    % For all images
    for image = 1:size(x, 4)
        % Non-linear Mapping 1
        %x_nonlinear = x(:,:,1,image).^gamma;
        
        % Non-linear Mapping 2
        x_nonlinear = log(x(:,:,1,image)+1);
        
        % No Non-linear Mapping Applied
        %x_nonlinear = x(:,:,1,image)
        
        % Getting Horizontal and Vertical Gradients
        [Gx, Gy] = imgradientxy(x_nonlinear);
        % Getting Magnitude and Direction of H and V Gradients
        % Angle lies between [-pi, pi]
        [Gmag, Gdir] = imgradient(Gx, Gy);
        %[Gmag, Gdir] = imgradient(x_nonlinear);
        
        % Store features of each image, then concatenate it with other image features to form a single matrix
        block_feature = [];
        
        % For the grid of binSize x binSize
        for i = 0:binSize-1
            for j = 0:binSize-1
                  
                % For each cell
                % We get the magnitudes and directions for that cell
                mag_patch = Gmag(4*i+1 : 4*i+7 , 4*j+1 : 4*j+7);
                ang_patch = Gdir(4*i+1 : 4*i+7 , 4*j+1 : 4*j+7);
                
                % Store histogram for each cell here
                histr  = zeros(8,1);
                
                for p=1:7
                    for q=1:7
%                       
                        alpha= ang_patch(p,q);
                        mag = mag_patch(p,q);
                        
                        % Binning Process (Bi-Linear Interpolation)
                        % Assigning each pixel to the nearest orientation in numOri and with vote proportional to the gradient magnitude
                        if alpha>=-90 && alpha<-67.5
                            histr(1)=histr(1)+ mag*(-67.5-alpha)/22.5;
                            histr(2)=histr(2)+ mag*(alpha+90)/22.5;
                        elseif alpha>=-67.5 && alpha<-45
                            histr(2)=histr(2)+ mag*(-45-alpha)/22.5; 
                            histr(3)=histr(3)+ mag*(alpha+67.5)/22.5;
                        elseif alpha>=-45 && alpha<-22.5
                            histr(3)=histr(3)+ mag*(-22.5-alpha)/22.5; 
                            histr(4)=histr(4)+ mag*(alpha+45)/22.5;
                        elseif alpha>=-22.5 && alpha<0
                            histr(4)=histr(4)+ mag*(0-alpha)/22.5; 
                            histr(5)=histr(5)+ mag*(alpha+22.5)/22.5;
                        elseif alpha>=0 && alpha<22.5
                            histr(5)=histr(5)+ mag*(22.5-alpha)/22.5; 
                            histr(6)=histr(6)+ mag*(alpha-0)/22.5;
                        elseif alpha>=22.5 && alpha<45
                            histr(6)=histr(6)+ mag*(45-alpha)/22.5; 
                            histr(7)=histr(7)+ mag*(alpha-22.5)/22.5;
                        elseif alpha>=45 && alpha<67.5
                            histr(7)=histr(7)+ mag*(67.5-alpha)/22.5; 
                            histr(8)=histr(8)+ mag*(alpha-45)/22.5;
                        elseif alpha>=67.5 && alpha<90
                            histr(8)=histr(8)+ mag*(90-alpha)/22.5; 
                            histr(1)=histr(1)+ mag*(alpha-67.5)/22.5;
                        end
                                     
                    end
                end
                block_feature = [block_feature, histr];
                %block_feature = block_feature / sqrt(norm(block_feature)^2+.01);
                %features = block_feature;
            end
        end
        
        % Normalization of the feature vector using L2-Norm with a threshold
        block_feature=block_feature/sqrt(norm(block_feature)^2+.001);
        for z=1:length(block_feature)
            if block_feature(z)>0.2
                block_feature(z)=0.2;
            end
        end
        block_feature=block_feature/sqrt(norm(block_feature)^2+.001);
        
        block_feature = extractHOGFeatures(x(:,:,1,image), 'CellSize', [4,4], 'NumBins', 8);
        features = [features transpose(block_feature)];
    end
    
function features = lbpFeatures(x)
    bin = [1,2,4;8,0,16;32,64,128];
    features = [];
    for image = 1:size(x, 4)
        hist = zeros(256,1);
        im = x(:,:,1,image);
        for i = 1:size(im, 1)-2
            for j = 1:size(im,2)-2
                patch = im(i:i+3-1, j:j+3-1);
                patch = patch - patch(2,2);
                patch(patch < 0) = 0;
                patch(patch > 0) = 1;
                patch = patch .* bin;
                hist(sum(sum(patch, 1),2)+1) = hist(sum(sum(patch, 1),2)+1) + 1;               
            end
        end
        %feature = extractLBPFeatures(im);
        %features = [features transpose(feature)];
        %hist = sqrt(hist);
        hist = hist/norm(hist);
        features = [features hist];
    end
