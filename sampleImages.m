function [xlist ylist listofindexes] = sampleImages(I, imageDim, numImagesPerCategory, L, numCategories, seed, indexes)
% Randomly sample numCategories * numImagesPerCategory image patches of
% size imageDim x imageDim from the image I. L is the 2D label matrix, 
% numCategories is an integer with the number of categories, seed (optional) 
% is the random generator seed, and indexes (optional) is a vector of
% indexes for center pixels to be selected instead of random extraction.
%
% OUTPUT:
% images: 4D matrix of image patches
% labels: 1-M vector of labels for the middle pixel of each image patch
% listofindexes: list of selected indexes

imageChannels = size(I,3);
halfImg = floor(imageDim/2);

if nargin==7
    listofindexes = indexes;
else
    
    if nargin<6
        rand('state', 0)
    else
        rand('state', seed)
    end   
    
    % Reset borders so they dont get selected
    L(1:imageDim,:)=0;
    L(end-imageDim:end,:)=0;
    L(:,1:imageDim)=0;
    L(:,end-imageDim:end,:)=0;
    
    % Randomly select numImagesPerCategory
    categories = 1:numCategories;
    listofindexes = [];
    for i=categories
%         listofindexes = [listofindexes; datasample(find(L==i), numImagesPerCategory(i), 'Replace', false)];
         listofindexes = [listofindexes; datasample(find(L==i), numImagesPerCategory(i))];
    end
end

% Extract patches
[xlist,ylist] = ind2sub(size(I),listofindexes);

% images = zeros(imageDim, imageDim, imageChannels, numel(xlist),'single');
% labels = zeros(1,numel(xlist), 'double');
% for i=1:numel(xlist)
%     x = xlist(i);
%     y = ylist(i);
%     im = 1;
%     images(:,:,:,i) = I(x - halfImg: x + halfImg, y - halfImg: y + halfImg, :, im);
%     labels(i) = L(x,y);
% end
% 

end