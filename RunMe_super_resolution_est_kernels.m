%%  IDBP-CNN for super-resolution with estimated kernel, with and without image-adaptaion

% Reference: "Super-Resolution via Image-Adapted Denoising CNNs: Incorporating External and Internal Learning"
% Authors: Tom Tirer and Raja Giryes
% Journal: IEEE Signal Processing Letters, 2019.


clear;
addpath('utilities');
useGPU = 1; % disable it if you do not have GPU support for MatConvNet

flag_kernel_is_unknown = 1;
% If you assume that the kernels are unknown, you need to perform kernel estimation.
% It seems to be quite hard to find online code for kernel estimation for SR. 
% Therefore, we downloaded a method for deblurring from "http://zoi.utia.cas.cz/files/caip2013_code_0.zip",
% and adjusted the estimated kernel for SR in our code.
addpath('kernel_estimator\caip2013_code_0');
% If you choose to use this kernel estimation make sure that:
% 1. in "caip2013_code_0\parameters.m" set "PAR.verbose = 0;", to avoid displaying logs are debug figures
% 2. FIX: in "caip2013_code_0\PSFestimaLnoRgrad.m" line 70 
%         replace:    eG = edgetaper(G,ones(hsize)/prod(hsize));
%         with:    if hsize>1; eG = edgetaper(G,ones(hsize)/prod(hsize)); else; eG = G; end;
% 3. FIX: in "caip2013_code_0\fftCGSRaL.m" line 74
%         replace:    eG = edgetaper(G,H);
%         with:    eG = edgetaper(G,H+eps);
% 4. cite the relevant paper which is mentioned in "http://zoi.utia.cas.cz/deconv_sparsegrad"


flag_use_IA_denoiser = 0; % 0: use IDBP-CNN, 1: use IDBP-CNN-IA 
flag_IA_also_final_denoiser = 0; % if equals 0 (and: flag_use_IA_denoiser=1) only the penultimate denoiser will be image-adapted
flag_imshow_est_SR_images = 1;
dataset_choice_index = 1;

dataset_options = {'Set5','Set14'};

s = 2; % SR scale factor
N_scenarios = 8; % scenario for each different kernel
kernels_folder = 'test_sets\estimated_kernels';

run('D:\Documents_D\MATLAB\matconvnet-1.0-beta25\matlab\vl_setupnn.m'); % put the path which is relevant to your computer
load('Denoisers_folder\IRCNN\models\modelgray.mat'); % loads CNNdenoiser from IRCNN offline training


dataset_choice = dataset_options{dataset_choice_index};
images_folder = ['test_sets\' dataset_choice];
ext                 =  {'*.jpg','*.png','*.bmp'};
images_list           =  [];
for i = 1 : length(ext)
    images_list = cat(1,images_list,dir(fullfile(images_folder, ext{i})));
end
N_images = size(images_list,1);


all_results_PSNR = zeros(N_scenarios,N_images);
all_results_ssim = zeros(N_scenarios,N_images);
PSNR_array_all = [];

for scenario_ind=1:1:N_scenarios
    for image_ind=1:1:N_images
        
        %% prepare observations
        g = gpuDevice(1);
        reset(g);
        
        img_name = images_list(image_ind).name;
        X0 = imread(fullfile(images_folder,img_name));
        
        assert(max(X0(:))<=255);
            
        [M,N,~] = size(X0);
        X0 = double(X0);
        
        % load true kernel
        h = load(fullfile('test_sets\non_ideal_kernels',['True_Kernel_h_' num2str(scenario_ind) '.mat']));
        h = h.h;
        h0 = h;

        Hfunc = @(Z) downsample2(imfilter(Z,h,'conv','replicate'),s);
        Htfunc = @(Z) imfilter(upsample2_MN(Z,s,[M,N]),fliplr(flipud(conj(h))),'conv','replicate');     
            
        Y_clean = [];
        for c=1:1:size(X0,3)
            Y_clean = cat(3,Y_clean,Hfunc(X0(:,:,c)));
        end
        [Mlr,Nlr] = size(Y_clean(:,:,1));
        
        randn('seed',0);
        sig_e = 0;
        noise = sig_e * randn(size(Y_clean));
        Y = Y_clean + noise;
        
        if flag_kernel_is_unknown
            % perform kernel estimation from the LR image Y
            % the estimated kernel will be used in the algorithm instead of the true kernel
            [U, H, report] = MCrestoration(Y,[15,15]);
            h = H(2:end,2:end);
            [Xq,Yq] = meshgrid(-7:1:7,-7:1:7);
            [Xc,Yc] = meshgrid(-7:0.5:7,-7:0.5:7);
            h = interp2(Xq,Yq,h,Xc,Yc,'bicubic');
            
            % The following steps improve the resulted h, done also for all competitors
            h(h<max(h(:))*0.15)=0;
            % centralize estimated kernel
            temp = imfilter(h,ones(1));
            [~,cen_ind] = max(temp(:));
            [cen_row, cen_col] = ind2sub(size(h),cen_ind);
            h_width = min([cen_row-1, size(h,1)-cen_row, cen_col-1, size(h,2)-cen_col]);
            h = h(cen_row-h_width:cen_row+h_width,cen_col-h_width:cen_col+h_width);

            h = h/sum(h(:));          
            Hfunc = @(Z) downsample2(imfilter(Z,h,'conv','replicate'),s);
            Htfunc = @(Z) imfilter(upsample2_MN(Z,s,[M,N]),fliplr(flipud(conj(h))),'conv','replicate');
        end
                
        clear X_tilde_ycbcr;
        if size(X0,3)==3
            X0_rgb = X0; Y_rgb = Y; % saved for (possible) display at the end of alg
            X0_ycbcr = rgb2ycbcr(X0/255)*255;
            X0 = X0_ycbcr(:,:,1); % for computing PSNR in Y channel as done in benchmarks
            X_bicubic = imresize(Y,[M,N],'bicubic');
            X_tilde_ycbcr = rgb2ycbcr(X_bicubic/255)*255;            
            Y = rgb2ycbcr(double(Y/255))*255;
            Y = Y(:,:,1);
            Y_upscaled = X_tilde_ycbcr(:,:,1);
            if flag_kernel_is_unknown
                U = rgb2ycbcr(U/255)*255;
                U = U(:,:,1); % to be used for IA
            end
        else
            Y_upscaled = imresize(Y,[M,N],'bicubic');
        end
        input_PSNR = 10*log10(255^2/mean((X0(:)-Y_upscaled(:)).^2))        


        %% make the penultimate CNN denoiser image adaptive (IA)
        
        if flag_use_IA_denoiser
            
            delta = s+1;
            sigma_alg = sig_e + delta;
            sigma_alg_ft = sigma_alg;
            
            net = loadmodel(sigma_alg,CNNdenoiser);
            
            if flag_kernel_is_unknown
                net_ft = fine_tune_CNN_denoiser(U,sigma_alg,net,useGPU);
            else
                net_ft = fine_tune_CNN_denoiser(Y,sigma_alg,net,useGPU);
            end
            
        end
        
        %% run IDBP super-resolution
        
        maxIter = 30;
        use_different_delta = 1; delta = 13;
        delta_list = 0 + logspace(log10(12*s),log10(s),maxIter);
        epsilon = 0.045; % regularization for conjugate gradients, e.g. use 0.04-0.05 for non-ideal or estimated kernels
        sigma_stop_decreasing_delta = 0; % use large value (e.g. 10) it if the Y is degraded
        
        % initialization
        Z_tilde = Y_upscaled; X_tilde = Y_upscaled;
        sigma_alg = sig_e + delta;
        HHt_cg = @(z) vec(Hfunc(Htfunc(reshape(z,Mlr,Nlr))))+z*epsilon;
        
        PSNR_array = [];
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
                Z_tilde_c(Z_tilde_c<0)=0; Z_tilde_c(Z_tilde_c>255)=255;
                Z_tilde = cat(3,Z_tilde,Z_tilde_c);
            end
            
            % compute X_tilde
            if ( ~flag_use_IA_denoiser || sigma_alg>=(sigma_alg_ft+1))
                X_tilde = CNN_denoising_operation(Z_tilde,sigma_alg,CNNdenoiser,useGPU);
                if max(X_tilde(:))<=1.5; X_tilde = X_tilde*255; end;
                
            else % use adaptive fine-tuned denoiser
                disp(['uses IA-CNN denoiser: sigma_alg_ft = ' num2str(sigma_alg_ft)]);
                
                if sigma_alg<=(s+0.5) && flag_fine_tune_final_CNN_denoiser
                    flag_fine_tune_final_CNN_denoiser = 0;
                    net = loadmodel(s,CNNdenoiser);
                    if flag_kernel_is_unknown
                        net_ft = fine_tune_CNN_denoiser(U,s,net,useGPU);
                    else
                        net_ft = fine_tune_CNN_denoiser(Y,s,net,useGPU);
                    end
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
            
            % compute PSNR
            % since the estimated kernel may be shifted - alignment is required
            [n,m,ch]=size(X0);
            row = ceil(s); col = ceil(s);
            if sig_e == 0
                X_est = double(Z_tilde(row+1:n-row,col+1:m-col,:)); % crop edges like most methods, e.g. IRCNN
            else
                X_est = double(X_tilde(row+1:n-row,col+1:m-col,:)); % crop edges like most methods, e.g. IRCNN
            end
            TM = vision.TemplateMatcher('Metric','Sum of squared differences'); % ignoring translations
            loc = double(step(TM,X0,X_est));
            X0_ = X0( loc(2)-floor((size(X_est,1)-1)/2):loc(2)+ceil((size(X_est,1)-1)/2), loc(1)-floor((size(X_est,2)-1)/2):loc(1)+ceil((size(X_est,2)-1)/2));
            [output, Greg] = dftregistration(fft2(X0_),fft2(X_est),64);
            X_est = real(ifft2(Greg));
            X_est = X_est(4:end-3,4:end-3);
            X0_ = X0_(4:end-3,4:end-3);
            
            X_est_clip = X_est; X_est_clip(X_est<0) = 0; X_est_clip(X_est>255) = 255;
            PSNR = 10*log10(255^2/mean((X0_(:)-X_est_clip(:)).^2));
            
            disp(['IDBP: finished iteration ' num2str(k) ', PSNR for X_tilde = ' num2str(PSNR) ', sigma_alg = ' num2str(sigma_alg)]);
            PSNR_array = [PSNR_array, PSNR];
            
        end
        
        if sig_e == 0  && sigma_stop_decreasing_delta <= 0
            % in the noiseless case, take the last Z_tilde as the estimation
            
            % compute Z_tilde - often reduces performance for non IA
            if flag_use_IA_denoiser
                Z_tilde = [];
                for c=1:1:size(X_tilde,3)
                    H_X_tilde = Hfunc(X_tilde(:,:,c));
                    [cg_result, iter, residual] = cg(zeros(Mlr*Nlr,1), HHt_cg, vec(Y(:,:,c)-H_X_tilde), 100, 10^-6); % cg_result = inv(H*Ht)*(Y-H*X_tilde)
                    if sqrt(residual)>10^-3
                        disp(['cg: finished after ' num2str(iter) ' iterations with norm(residual) = ' num2str(sqrt(residual)) ' - Use preconditioning or tikho regularization (epsilon) for HHt_cg']);
                    end
                    Z_tilde_c = Htfunc(reshape(cg_result,Mlr,Nlr)) + X_tilde(:,:,c);
                    Z_tilde_c(Z_tilde_c<0)=0; Z_tilde_c(Z_tilde_c>255)=255;
                    Z_tilde = cat(3,Z_tilde,Z_tilde_c);
                end
            end            
            X_tilde = Z_tilde;
        end
        PSNR_array_all = [PSNR_array_all; PSNR_array];
        
        
        if exist('X_tilde_ycbcr','var')
            X_tilde_ycbcr(:,:,1) = X_tilde;
            X_tilde_color  = ycbcr2rgb(X_tilde_ycbcr/255)*255;
            if flag_imshow_est_SR_images; figure; imshow(uint8(X_tilde_color)); end;
        else
            if flag_imshow_est_SR_images; figure; imshow(uint8(X_tilde)); end;
        end        
        
        %% collect results
        
        % since the estimated kernel may be shifted - alignment is required
        [n,m,ch]=size(X0);
        row = ceil(s); col = ceil(s);
        X_est = double(X_tilde(row+1:n-row,col+1:m-col,:)); % crop edges like most methods, e.g. IRCNN.
        TM = vision.TemplateMatcher('Metric','Sum of squared differences'); % ignoring translations
        loc = double(step(TM,X0,X_est));
        X0_ = X0( loc(2)-floor((size(X_est,1)-1)/2):loc(2)+ceil((size(X_est,1)-1)/2), loc(1)-floor((size(X_est,2)-1)/2):loc(1)+ceil((size(X_est,2)-1)/2));
        [output, Greg] = dftregistration(fft2(X0_),fft2(X_est),64);
        X_est = real(ifft2(Greg));
        X_est = X_est(4:end-3,4:end-3);
        X0_ = X0_(4:end-3,4:end-3);
        
        X_est_clip = X_est; X_est_clip(X_est<0) = 0; X_est_clip(X_est>255) = 255;
        PSNR = 10*log10(255^2/mean((X0_(:)-X_est_clip(:)).^2));
        
        all_results_PSNR(scenario_ind,image_ind) = PSNR;
        ssim_res = ssim(double(X_est_clip)/255,double(X0_)/255); % we use MATLAB R2016a function
        all_results_ssim(scenario_ind,image_ind) = ssim_res;
        disp(['flag_use_IA=' num2str(flag_use_IA_denoiser) ', scenario_ind=' num2str(scenario_ind)  ', image_ind=' num2str(image_ind) ', PSNR=' num2str(PSNR) ', SSIM=' num2str(ssim_res)]);

    end
end

figure; plot(mean(PSNR_array_all,1));
grid on; xlabel('SNR [dB]'); xlabel('Iteration'); ylabel('PSNR [dB]');
disp(['Average PSNR: ' num2str(mean(all_results_PSNR(:)))]);


