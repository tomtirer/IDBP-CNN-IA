function [theta_update,next_mean,next_var] = adam_update(t,curr_grad,curr_mean,curr_var,rho)

beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

next_mean = beta1*curr_mean + (1-beta1)*curr_grad;
next_var = beta2*curr_var + (1-beta2)*curr_grad.^2;
alpha = rho * sqrt(1-beta2^t)/(1-beta1^t);

theta_update = - alpha * next_mean./(next_var.^0.5 + epsilon);

