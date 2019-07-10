function X_tilde = CNN_denoising_operation(Y_tilde,sigma,CNNdenoiser,useGPU)

net = loadmodel(sigma,CNNdenoiser);
net = vl_simplenn_tidy(net);
if useGPU
    Y_tilde = gpuArray(Y_tilde);
    net = vl_simplenn_move(net, 'gpu');
end
res = vl_simplenn(net,single(Y_tilde/255),[],[],'conserveMemory',true,'mode','test');
X_tilde = Y_tilde - 255*res(end).x;
if useGPU
    X_tilde = gather(X_tilde);
end

