function [net]=train_multi(imdb)

%
run ('vl_setupnn.m');

%%%%%common options%%%%%%
trainOpts.batchSize=10;
trainOpts.numEpochs=100;
trainOpts.continue=true;
trainOpts.gpus=[];
trainOpts.learningRate=0.001;
trainOpts.expDir='data/test_pu' ;
trainOpts.numSubBatches=1;

%getBatch options

bopts.useGpu=numel(trainOpts.gpus)>0;

%%%%network definition

net=dagnn.DagNN();

%% for input_1
net.addLayer('conv1',dagnn.Conv('size',[2 2 103 200], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'input_1'}, {'conv1'}, {'conv1f'  'conv1b'});
net.addLayer('pool1', dagnn.Pooling ('method', 'max', 'poolSize', [ 2, 2],...
    'stride', [1 1 ], 'pad', [0 0 0 0 ]), {'conv1'}, {'pool1'},{});
net.addLayer('conv2',dagnn.ConvTranspose('size',[2 2 200 200], 'hasBias', 'true'),{'pool1'}, {'conv2'}, {'conv2f'  'conv2b'});
net.addLayer('relu1',dagnn.ReLU(),{'conv2'},{'relu1'},{});

%% for input_2

net.addLayer('conv3',dagnn.Conv('size',[2 2 20 200], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'input_2'}, {'conv3'}, {'conv3f'  'conv3b'});
net.addLayer('pool2', dagnn.Pooling ('method', 'max', 'poolSize', [ 2, 2],...
    'stride', [1 1 ], 'pad', [0 0 0 0 ]), {'conv3'}, {'pool2'},{});
net.addLayer('conv4',dagnn.ConvTranspose('size',[2 2 200 200], 'hasBias', 'true'),{'pool2'}, {'conv4'}, {'conv4f'  'conv4b'});
net.addLayer('relu2',dagnn.ReLU(),{'conv4'},{'relu2'},{});

%% for input_3
net.addLayer('conv5',dagnn.Conv('size',[2 2 28 200], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'input_3'}, {'conv5'}, {'conv5f'  'conv5b'});
net.addLayer('pool3', dagnn.Pooling ('method', 'max', 'poolSize', [ 2, 2],...
    'stride', [1 1 ], 'pad', [0 0 0 0 ]), {'conv5'}, {'pool3'},{});
net.addLayer('conv6',dagnn.ConvTranspose('size',[2 2 200 200], 'hasBias', 'true'),{'pool3'}, {'conv6'}, {'conv6f'  'conv6b'});
net.addLayer('relu3',dagnn.ReLU(),{'conv6'},{'relu3'},{});

%% for input_4
net.addLayer('conv7',dagnn.Conv('size',[2 2 12 200], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'input_4'}, {'conv7'}, {'conv7f'  'conv7b'});
net.addLayer('pool4', dagnn.Pooling ('method', 'max', 'poolSize', [ 2, 2],...
    'stride', [1 1 ], 'pad', [0 0 0 0 ]), {'conv7'}, {'pool4'},{});
net.addLayer('conv8',dagnn.ConvTranspose('size',[2 2 200 200], 'hasBias', 'true'),{'pool4'}, {'conv8'}, {'conv8f'  'conv8b'});
net.addLayer('relu4',dagnn.ReLU(),{'conv8'},{'relu4'},{});

%% for input_5
net.addLayer('conv9',dagnn.Conv('size',[2 2 12 200], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'input_5'}, {'conv9'}, {'conv9f'  'conv9b'});
net.addLayer('pool5', dagnn.Pooling ('method', 'max', 'poolSize', [ 2, 2],...
    'stride', [1 1 ], 'pad', [0 0 0 0 ]), {'conv9'}, {'pool5'},{});
net.addLayer('conv10',dagnn.ConvTranspose('size',[2 2 200 200], 'hasBias', 'true'),{'pool5'}, {'conv10'}, {'conv10f'  'conv10b'});
net.addLayer('relu5',dagnn.ReLU(),{'conv10'},{'relu5'},{});

%% concatenate the features for different inputs
net.addLayer('concat', dagnn.Concat('dim', 2), {'relu1', 'relu2','relu3','relu4','relu5'}, {'concat'}); 

net.addLayer('classifier',dagnn.Conv('size',[4 20 200 9], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'concat'}, {'classifier'}, {'conv11f'  'conv11b'});
net.addLayer('prediction',dagnn.SoftMax(),{'classifier'},{'prediction'},{});
net.addLayer('objective',dagnn.Loss('loss', 'log'), {'prediction','label'},{'objective'},{});
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prediction', 'label'},'error');

%%%%%   end of the network   %%%%%%%%%%%%%%

%%%%%%% initialization of the weights%%%%%%%%
initNet(net, 1/100);

%%%%%%%%%%%  training %%%%%%%%%%
rng('default')
net=cnn_train_dag(net,imdb, @getBatchA,trainOpts);
end

function initNet(net,f)
net.initParams();
%%%%if is a convolution layer%%%%%%%%%
for l=1:length(net.layers)
    if(strcmp(class(net.layers(l).block),'dagnn.Conv'))
        f_ind=net.layers(l).paramIndexes(1);
        b_ind=net.layers(1).paramIndexes(2);
        
        
        net.params(f_ind).value=f*randn(size(net.params(f_ind).value), 'single');
        net.params(f_ind).learningRate=1;
        net.params(f_ind).weightDecay=1;
        
        
        net.params(b_ind).value=f*randn(size(net.params(b_ind).value), 'single');
        net.params(b_ind).learningRate=2;
        net.params(b_ind).weightDecay=1;

    end
    
end
end



%%%%%%%%%%%%function on charge of creating a batch of images+labels
function inputs=getBatchA(imdb, batch)

images_src_1 = imdb.images.data_src1(:,:,:,batch) ;
images_src_2 = imdb.images.data_src2(:,:,:,batch) ;
images_src_3 = imdb.images.data_src3(:,:,:,batch) ;
images_src_4 = imdb.images.data_src4(:,:,:,batch) ;
images_src_5 = imdb.images.data_src5(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5
    images_src_1 =fliplr(images_src_1 ) ; 
    images_src_2 =fliplr(images_src_2 ) ; 
    images_src_3 =fliplr(images_src_3 ) ; 
    images_src_4 =fliplr(images_src_4 ) ; 
    images_src_5 =fliplr(images_src_5 ) ;
end

% **********************************************
% Define the inputs cell-array to the DAG
% **********************************************
inputs = {'input_1', images_src_1 , 'label', labels, 'input_2', images_src_2, 'label', labels,...
    'input_3', images_src_3, 'label', labels,'input_4', images_src_4, 'label', labels,...
    'input_5', images_src_5, 'label', labels,} ;
end