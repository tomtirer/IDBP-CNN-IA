function net = fine_tune_CNN_denoiser(img_in,sig_noise,net,useGPU)

% Reference: "Super-Resolution via Image-Adapted Denoising CNNs: Incorporating External and Internal Learning"
% Authors: Tom Tirer and Raja Giryes
% Journal: IEEE Signal Processing Letters, 2019.

N_iter = 320;
miniBatchSize = 32;
update_CNN_params.N_layers_to_train = inf; % counted from the end, inf=all
noise_diff = 1;

update_CNN_params.learningRate = 3e-4;
update_CNN_params.t = 0;
update_CNN_params.weightDecay = 0;
update_CNN_params.run_adam = 1;
update_CNN_params.momentum = 0.9;

rng('default');
rng(0) ;

net.layers{end+1} = struct('type', 'pdist',...
                           'p',1,...
                           'noRoot', true) ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

if 1
  for i=1:numel(net.layers)
    J = numel(net.layers{i}.weights) ;
    if ~isfield(net.layers{i}, 'learningRate')
      net.layers{i}.learningRate = ones(1, J) ;
    end
    if ~isfield(net.layers{i}, 'weightDecay')
      net.layers{i}.weightDecay = ones(1, J) ;
    end
  end
end
% initialize with momentum 0
for i = 1:numel(net.layers)
    for j = 1:numel(net.layers{i}.weights)
        state.momentum{i}{j} = 0 ; % for SGD & ADAM
        state.variance{i}{j} = 0 ; % for ADAM
    end
end

err_per_epoch = [];


%% training

if useGPU
    net = vl_simplenn_move(net, 'gpu');
end

numInnerIter = 32;
N_epochs = round(N_iter/numInnerIter); % not really epochs... just numOuterIter
N_train_samples = miniBatchSize*numInnerIter;
t = update_CNN_params.t;

for epoch=1:1:N_epochs
    t_start_epoch = tic;
        
    start_ind = 1;
    sum_error_epoch = 0;
    sum_samples_epoch = 0;
    
    for jj=1:1:numInnerIter
        
        t = t+1;
        t_start = tic;
        finish_ind = min([start_ind + miniBatchSize - 1, N_train_samples]);
        
        gt_miniBatch = [];
        input_miniBatch = [];
        patch_size_opts = [34, 40, 50];
        patch_size = patch_size_opts(randi(length(patch_size_opts)));

        % prepare minibatch, use some data augmentation
        for ii=start_ind:1:finish_ind

            img = img_in;
            
            scales  = [1, 0.9];
            img = imresize(img,scales(randi(length(scales))));
            [M,N,~] = size(img);
            
            patch_center_col = randi(M-patch_size,1) + patch_size/2;
            patch_center_row = randi(N-patch_size,1) + patch_size/2;
            img_patch = img(patch_center_col-patch_size/2:patch_center_col+patch_size/2,patch_center_row-patch_size/2:patch_center_row+patch_size/2,:);
            
            N_sectors = 4;
            img_patch = rot90(img_patch,(randi(N_sectors)-1));
            
            if rand>0.5; img_patch = fliplr(img_patch); end;
            if rand>0.5; img_patch = flipud(img_patch); end;
            
            noise = (sig_noise+noise_diff) * randn(size(img_patch));
            y_patch = img_patch + noise;
            
            gt_miniBatch = cat(4,gt_miniBatch,img_patch);
            input_miniBatch = cat(4,input_miniBatch,y_patch);
            
        end
        
        gt_for_net = single(input_miniBatch-gt_miniBatch)/255; % residual learning
        %gt_for_net = single(gt_miniBatch)/255;
        input_for_net = single(input_miniBatch)/255;
        if useGPU
            gt_for_net = gpuArray(gt_for_net);
            input_for_net = gpuArray(input_for_net);
        end
        
        net.layers{end}.class = gt_for_net;
        
        % forward pass
        res = vl_simplenn(net,input_for_net,1,[],'conserveMemory',true,'mode','normal','SkipForward',false) ;
        
        sum_error = sum(squeeze(sum(sum(res(end).x,1),2))) / (patch_size+1)^2 ;
        sum_error_epoch = sum_error_epoch + sum_error;
        sum_samples_epoch = sum_samples_epoch + size(input_miniBatch,4);       
        
        % backward pass
        update_CNN_params.t = t;
        [net,state,net_diff_norm] = update_CNN(net,res,state,update_CNN_params);
        t_end = toc(t_start);
        
%         if mod(jj,numInnerIter)==0
%             disp(['Fine tuning DnCNN: finished iteration ' num2str((epoch-1)*numInnerIter+jj) ': norm(net_diff) = ' num2str(net_diff_norm) ', error = ' num2str(sum_error/size(input_miniBatch,4)) ', time elapsed = ' num2str(t_end)]);
%         end
        
        start_ind = start_ind + miniBatchSize;
    end
    err_per_epoch = [err_per_epoch, sum_error_epoch/sum_samples_epoch];
    t_end_epoch = toc(t_start_epoch);
    disp(['Fine tuning DnCNN: finished iteration ' num2str(epoch*numInnerIter) ', mean error = ' num2str(err_per_epoch(end)) ', time elapsed = ' num2str(t_end_epoch) ', sig_noise = ' num2str(sig_noise)]);
    
end

net.layers{end} = []; % remove loss layer
net.layers = net.layers(~cellfun('isempty', net.layers));

if useGPU
    net = vl_simplenn_move(net, 'cpu');
end



