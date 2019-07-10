%%  IDBP-CNN for super-resolution of old images, with and without image-adaptaion

% Reference: "Super-Resolution via Image-Adapted Denoising CNNs: Incorporating External and Internal Learning"
% Authors: Tom Tirer and Raja Giryes
% Journal: IEEE Signal Processing Letters, 2019.


clear;
addpath('utilities');
useGPU = 1; % disable it if you do not have GPU support for MatConvNet

flag_use_IA_denoiser = 1; % 0: use IDBP-CNN, 1: use IDBP-CNN-IA 
flag_IA_also_final_denoiser = 1; % if equals 0 (and: flag_use_IA_denoiser=1) only the penultimate denoiser will be image-adapted
flag_imshow_est_SR_images = 1;
dataset_choice = 'old7_LR';

s = 2; % SR scale factor
h = prepare_cubic_filter(1/s);


run('D:\Documents_D\MATLAB\matconvnet-1.0-beta25\matlab\vl_setupnn.m'); % put the path which is relevant to your computer
load('Denoisers_folder\IRCNN\models\modelgray.mat'); % loads CNNdenoiser from IRCNN offline training


images_folder = ['test_sets\' dataset_choice];
ext                 =  {'*.jpg','*.png','*.bmp'};
images_list           =  [];
for i = 1 : length(ext)
    images_list = cat(1,images_list,dir(fullfile(images_folder, ext{i})));
end
N_images = size(images_list,1);


for image_ind=1:1:N_images
    
    %% prepare observations
    g = gpuDevice(1);
    reset(g);
    
    img_name = images_list(image_ind).name;
    Y = imread(fullfile(images_folder,img_name));
    
    assert(max(Y(:))<=255);
    
    [Mlr,Nlr,~] = size(Y);
    M = Mlr*s; N=Nlr*s;
    Y = double(Y);   
    
    Hfunc = @(Z) downsample2(imfilter(Z,h,'conv','replicate'),s);
    Htfunc = @(Z) imfilter(upsample2_MN(Z,s,[M,N]),fliplr(flipud(conj(h))),'conv','replicate');
    
    randn('seed',0);
    
    clear X_tilde_ycbcr;
    if size(Y,3)==1 || sum(vec(abs(Y(:,:,1)-Y(:,:,2))))==0
        Y = Y(:,:,1);
        Y_upscaled = imresize(Y,[M,N],'bicubic');
    else
        Y_rgb = Y; % saved for (possible) display at the end of alg
        X_bicubic = imresize(Y,[M,N],'bicubic');
        X_tilde_ycbcr = rgb2ycbcr(X_bicubic/255)*255;
        Y = rgb2ycbcr(double(Y/255))*255;
        Y = Y(:,:,1);
        Y_upscaled = X_tilde_ycbcr(:,:,1);
    end
    
    
    %% make the penultimate CNN denoiser image adaptive (IA)
    
    if flag_use_IA_denoiser
        
        sigma_alg = 9; % uniform setting for the old images in the paper, you may try a different value to improve performance for other images
        sigma_alg_ft = sigma_alg;
        
        net = loadmodel(sigma_alg,CNNdenoiser);
        net_ft = fine_tune_CNN_denoiser(Y,sigma_alg,net,useGPU);
        
    end
    
    %% run IDBP super-resolution
        
    maxIter = 30;
    use_different_delta = 1; delta = 13;
    delta_list = 0 + logspace(log10(12*s),log10(s),maxIter);
    epsilon = 0; %  regularization for conjugate gradients, e.g. use 0.04-0.05 for non-ideal or estimated kernels
    sigma_stop_decreasing_delta = 10; % use large value (e.g. 10) it if the Y is degraded
    
    % initialization
    sig_e = 0;
    Z_tilde = Y_upscaled; X_tilde = Y_upscaled;
    sigma_alg = sig_e + delta;
    HHt_cg = @(z) vec(Hfunc(Htfunc(reshape(z,Mlr,Nlr))))+z*epsilon;
    
    k_stop_switching = [];
    flag_fine_tune_final_CNN_denoiser = flag_IA_also_final_denoiser;
    
    for k=1:1:maxIter
        
        if use_different_delta
            sigma_alg = sig_e + delta_list(k);
        end
        
        % compute Z_tilde
        Z_tilde = [];
        for c=1:1:size(X_tilde,3)
            H_X_tilde = Hfunc(X_tilde(:,:,c));
            [cg_result, iter, residual] = cg(zeros(Mlr*Nlr,1), HHt_cg, vec(Y(:,:,c)-H_X_tilde), 100, 10^-6); % cg_result = inv(H*Ht)*(Y-H*X_tilde)
            if sqrt(residual)>10^-3
                disp(['cg: finished after ' num2str(iter) ' iterations with norm(residual) = ' num2str(sqrt(residual)) ' - Use preconditioning or tikho regularization (epsilon) for HHt_cg']);
            end
            Z_tilde_c = Htfunc(reshape(cg_result,Mlr,Nlr)) + X_tilde(:,:,c);
            Z_tilde = cat(3,Z_tilde,Z_tilde_c);
        end
        
        % compute X_tilde
        if ( ~flag_use_IA_denoiser || sigma_alg>=(sigma_alg_ft+1))
            X_tilde = CNN_denoising_operation(Z_tilde,sigma_alg,CNNdenoiser,useGPU);
            if max(X_tilde(:))<=1.5; X_tilde = X_tilde*255; end;
            
        else % adaptive fine-tuned denoiser
            disp(['uses IA-CNN denoiser: sigma_alg_ft = ' num2str(sigma_alg_ft)]);
            
            if sigma_alg<=(s+0.5) && flag_fine_tune_final_CNN_denoiser
                flag_fine_tune_final_CNN_denoiser = 0;
                net = loadmodel(s,CNNdenoiser);
                net_ft = fine_tune_CNN_denoiser(Y,s,net,useGPU);
            end
            
            if useGPU
                Z_tilde = gpuArray(Z_tilde);
                net_ft = vl_simplenn_move(net_ft, 'gpu');
            end
            res = vl_simplenn(net_ft,single(Z_tilde/255),[],[],'conserveMemory',true,'mode','test');
            X_tilde = Z_tilde - 255*res(end).x;
            if useGPU
                Z_tilde = gather(Z_tilde);
                X_tilde = gather(X_tilde);
            end
        end
        
        if sigma_alg < sigma_stop_decreasing_delta
            use_different_delta = 0; % stop decreasing delta
            disp(['At k=' num2str(k) ': stop decreasing delta, sigma_alg = ' num2str(sigma_alg)]);
            if isempty(k_stop_switching); k_stop_switching = k; end;
        end
                
        disp(['IDBP: finished iteration ' num2str(k) ', sigma_alg = ' num2str(sigma_alg)]);
        
    end
    
    if sig_e == 0  && sigma_stop_decreasing_delta <= 0
        % in the noiseless case, take the last Z_tilde as the estimation
        
        % compute Z_tilde
        Z_tilde = [];
        for c=1:1:size(X_tilde,3)
            H_X_tilde = Hfunc(X_tilde(:,:,c));
            [cg_result, iter, residual] = cg(zeros(Mlr*Nlr,1), HHt_cg, vec(Y(:,:,c)-H_X_tilde), 100, 10^-6); % cg_result = inv(H*Ht)*(Y-H*X_tilde)
            if sqrt(residual)>10^-3
                disp(['cg: finished after ' num2str(iter) ' iterations with norm(residual) = ' num2str(sqrt(residual)) ' - Use preconditioning or tikho regularization (epsilon) for HHt_cg']);
            end
            Z_tilde_c = Htfunc(reshape(cg_result,Mlr,Nlr)) + X_tilde(:,:,c);
            Z_tilde = cat(3,Z_tilde,Z_tilde_c);
        end
        
        X_tilde = Z_tilde;
        
        disp(['IDBP (noiseless case): finished iteration ' num2str(k)]);
    end
    
    
    if exist('X_tilde_ycbcr','var')
        X_tilde_ycbcr(:,:,1) = X_tilde;
        X_tilde_color  = ycbcr2rgb(X_tilde_ycbcr/255)*255;
        if flag_imshow_est_SR_images; figure; imshow(uint8(X_tilde_color)); end;
    else
        if flag_imshow_est_SR_images; figure; imshow(uint8(X_tilde)); end;
    end
    
    %% collect results
    
%     [n,m,ch]=size(X0);
%     row = ceil(s); col = ceil(s);
%     X_est = double(X_tilde(row+1:n-row,col+1:m-col,:)); % crop edges like most methods, e.g. IRCNN.
%     TM = vision.TemplateMatcher('Metric','Sum of squared differences'); % ignoring translations
%     loc = double(step(TM,X0,X_est));
%     X0_ = X0( loc(2)-floor((size(X_est,1)-1)/2):loc(2)+ceil((size(X_est,1)-1)/2), loc(1)-floor((size(X_est,2)-1)/2):loc(1)+ceil((size(X_est,2)-1)/2));
%     
%     X_est_clip = X_est; X_est_clip(X_est<0) = 0; X_est_clip(X_est>255) = 255;
%     PSNR = 10*log10(255^2/mean((X0_(:)-X_est_clip(:)).^2));
    
    disp(['flag_use_IA=' num2str(flag_use_IA_denoiser) ', image_ind=' num2str(image_ind)]);
    
end


