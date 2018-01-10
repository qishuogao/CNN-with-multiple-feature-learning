function [dataset]=generate_dataset(Itrain_1,Itrain_2,Itrain_3,Itrain_4,Itrain_5,groundtruth,patch,xlist,ylist)

%% generate image_src_1

L=groundtruth;
halfImg = floor(patch/2);

channel_1=size(Itrain_1,3);
selectedChannels_1 = 1:channel_1; % Selected color channels
imageChannels_1 = length(selectedChannels_1);
train_images_1 = zeros(patch, patch, imageChannels_1, numel(xlist),'single');


channel_2=size(Itrain_2,3);
selectedChannels_2 = 1:channel_2; % Selected color channels
imageChannels_2 = length(selectedChannels_2);
train_images_2 = zeros(patch, patch, imageChannels_2, numel(xlist),'single');

channel_3=size(Itrain_3,3);
selectedChannels_3 = 1:channel_3; % Selected color channels
imageChannels_3 = length(selectedChannels_3);
train_images_3 = zeros(patch, patch, imageChannels_3, numel(xlist),'single');

channel_4=size(Itrain_4,3);
selectedChannels_4 = 1:channel_4; % Selected color channels
imageChannels_4 = length(selectedChannels_4);
train_images_4 = zeros(patch, patch, imageChannels_4, numel(xlist),'single');

channel_5=size(Itrain_5,3);
selectedChannels_5 = 1:channel_5; % Selected color channels
imageChannels_5 = length(selectedChannels_5);
train_images_5 = zeros(patch, patch, imageChannels_5, numel(xlist),'single');

for i=1:numel(xlist)
    x = xlist(i);
    y = ylist(i);
    im = 1;
    train_images_1(:,:,:,i) = Itrain_1(x - halfImg: x + halfImg, y - halfImg: y + halfImg, :, im);
    
    train_images_2(:,:,:,i) = Itrain_2(x - halfImg: x + halfImg, y - halfImg: y + halfImg, :, im);
    
    train_images_3(:,:,:,i) = Itrain_3(x - halfImg: x + halfImg, y - halfImg: y + halfImg, :, im);
    
    train_images_4(:,:,:,i) = Itrain_4(x - halfImg: x + halfImg, y - halfImg: y + halfImg, :, im);
    
    train_images_5(:,:,:,i) = Itrain_5(x - halfImg: x + halfImg, y - halfImg: y + halfImg, :, im);
    
    labels(i) = L(x,y);
end

images_1 = reshape(train_images_1,[],size(train_images_1,4));

images_2 = reshape(train_images_2,[],size(train_images_2,4));

images_3 = reshape(train_images_3,[],size(train_images_3,4));

images_4 = reshape(train_images_4,[],size(train_images_4,4));

images_5 = reshape(train_images_5,[],size(train_images_5,4));

% Split into training and validation sets
seed = 10;

[trainimages_1,valimages_1] = split(images_1,[0.9 0.1]);

[trainimages_2,valimages_2] = split(images_2,[0.9 0.1]);

[trainimages_3,valimages_3] = split(images_3,[0.9 0.1]);

[trainimages_4,valimages_4] = split(images_4,[0.9 0.1]);

[trainimages_5,valimages_5] = split(images_5,[0.9 0.1]);

[trainlabels,vallabels] = split(labels,[0.9 0.1]);% corresponding labels
% Reshape to 4D: dim x dim x channels x N

im_patches_4d_1=[trainimages_1 valimages_1];
im_patches_4d_1=reshape(im_patches_4d_1, patch, patch, imageChannels_1, []);


im_patches_4d_2=[trainimages_2 valimages_2];
im_patches_4d_2=reshape(im_patches_4d_2, patch, patch, imageChannels_2, []);

im_patches_4d_3=[trainimages_3 valimages_3];
im_patches_4d_3=reshape(im_patches_4d_3, patch, patch, imageChannels_3, []);

im_patches_4d_4=[trainimages_4 valimages_4];
im_patches_4d_4=reshape(im_patches_4d_4, patch, patch, imageChannels_4, []);

im_patches_4d_5=[trainimages_5 valimages_5];
im_patches_4d_5=reshape(im_patches_4d_5, patch, patch, imageChannels_5, []);

dataset.images.id=zeros(1,size(images_1,2));

dataset.images.id=1:size(labels,2);
% 4d image patches
dataset.images.data_src1=im_patches_4d_1;

dataset.images.data_src2=im_patches_4d_2;

dataset.images.data_src3=im_patches_4d_3;

dataset.images.data_src4=im_patches_4d_4;

dataset.images.data_src5=im_patches_4d_5;
% which set: training-1,validation-2
dataset.images.set=zeros(1,size(images_1,2));
dataset.images.set(1:size(trainimages_1,2))=1;
dataset.images.set(size(trainimages_1,2)+1:size(trainimages_1,2)+size(valimages_1,2))=2;
dataset.images.labels=[trainlabels vallabels];

