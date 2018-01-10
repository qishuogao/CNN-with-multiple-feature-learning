clc
clear all
close all

rng('default')
setup()

load PU
load PUEMAP
load Indexes
patch = 5;
numClasses = 9;


%%generate the dataset 1
img_1=pavia_corrected;
Itrain_1=mat2gray(img_1);
Itrain_1=single(Itrain_1);
% take average mean of each channel
mean_Itrain_1=mean(mean(double(Itrain_1),1),2);
for i=1:size(Itrain_1,3)
    Itrain_1(:,:,i)=Itrain_1(:,:,i)-mean_Itrain_1(1,1,i);
end


%%generate the dataset 2

img_2=PUarea;

Itrain_2=mat2gray(img_2);
Itrain_2=single(Itrain_2);
% take average mean of each channel
mean_Itrain_2=mean(mean(double(Itrain_2),1),2);
for i=1:size(Itrain_2,3)
    Itrain_2(:,:,i)=Itrain_2(:,:,i)-mean_Itrain_2(1,1,i);
end

%% genearte the dataset 3
img_3=PUdiag;

Itrain_3=mat2gray(img_3);
Itrain_3=single(Itrain_3);
% take average mean of each channel
mean_Itrain_3=mean(mean(double(Itrain_3),1),2);
for i=1:size(Itrain_3,3)
    Itrain_3(:,:,i)=Itrain_3(:,:,i)-mean_Itrain_3(1,1,i);
end
%% generate the dataset 4

img_4=PUinteria;

Itrain_4=mat2gray(img_4);
Itrain_4=single(Itrain_4);
% take average mean of each channel
mean_Itrain_4=mean(mean(double(Itrain_4),1),2);
for i=1:size(Itrain_4,3)
    Itrain_4(:,:,i)=Itrain_4(:,:,i)-mean_Itrain_4(1,1,i);
end
%% generate the dataset 5

img_5=PUstand;

Itrain_5=mat2gray(img_5);
Itrain_5=single(Itrain_5);
% take average mean of each channel
mean_Itrain_5=mean(mean(double(Itrain_5),1),2);
for i=1:size(Itrain_5,3)
    Itrain_5(:,:,i)=Itrain_5(:,:,i)-mean_Itrain_5(1,1,i);
end

%% generate imdb for network

[dataset]=generate_dataset(Itrain_1,Itrain_2,Itrain_3,Itrain_4,Itrain_5,groundtruth,patch,xlist,ylist);

% id of patches, id is the number of pixels in images



net=train_multi(dataset);
s=[];
dim=floor(patch/2);
Itest_1=padarray(Itrain_1,[dim,dim],'symmetric','both');
Itest_2=padarray(Itrain_2,[dim,dim],'symmetric','both');
Itest_3=padarray(Itrain_3,[dim,dim],'symmetric','both');
Itest_4=padarray(Itrain_4,[dim,dim],'symmetric','both');
Itest_5=padarray(Itrain_5,[dim,dim],'symmetric','both');
result_map=zeros(610,340);
score=zeros(640,340);
for i=1:207400
    [X,Y]=ind2sub(size(groundtruth),i);
    X_new = X+dim;
    Y_new = Y+dim;         
    X_range = X_new-dim : X_new+dim;
    Y_range = Y_new-dim : Y_new+dim;
    temp_1=Itest_1(X_range,Y_range,:);
    temp_2=Itest_2(X_range,Y_range,:);
    temp_3=Itest_3(X_range,Y_range,:);
    temp_4=Itest_4(X_range,Y_range,:);
    temp_5=Itest_5(X_range,Y_range,:);
    net.conserveMemory=0;
    net.eval({'input_1',temp_1,'input_2', temp_2,'input_3',temp_3,'input_4',temp_4,'input_5',temp_5});

%obtain the CNN output

    res=net.vars(net.getVarIndex('prediction')).value;
    scores=squeeze(gather(res));

%%%%%%show the classification results
    s=[s scores];
    [score(X,Y),result_map(X,Y)]=max(scores);
%       
   
end
[acc] = ComputeClassificationAccuracy(result_map,groundtruth)
prob=s; %% the probabalities for all samples
prob_train=s(:,indexes); %% the probalitiles for training samples

% save('ip_ori.mat','prob_ori','prob_ori_train','result_map','acc')