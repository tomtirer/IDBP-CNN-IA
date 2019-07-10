function [net,state,net_diff_norm] = update_CNN(net,res,state,update_CNN_params)

learningRate = update_CNN_params.learningRate;
t = update_CNN_params.t;
weightDecay = update_CNN_params.weightDecay;
run_adam = update_CNN_params.run_adam;
momentum = update_CNN_params.momentum ;
N_layers_to_train = inf; if isfield(update_CNN_params,'N_layers_to_train'); N_layers_to_train = update_CNN_params.N_layers_to_train; end;


batchSize = size(res(end).x,4);

net_diff_norm = 0;
N_layers_to_train_counter = 0;
for l=numel(net.layers):-1:1
    if N_layers_to_train_counter >= N_layers_to_train
        break;
    elseif strcmp(net.layers{l}.type, 'conv')
        N_layers_to_train_counter = N_layers_to_train_counter+1;
    end
    
    for j=numel(res(l).dzdw):-1:1
        parDer = res(l).dzdw{j}  ;
        
        if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
            % special case for learning bnorm moments
            thisLR = net.layers{l}.learningRate(j) ;
            net.layers{l}.weights{j} = vl_taccum(...
                1 - thisLR, ...
                net.layers{l}.weights{j}, ...
                thisLR / batchSize, ...
                parDer) ;
        else 
            % Standard gradient training.
            thisDecay = weightDecay * net.layers{l}.weightDecay(j) ;
            thisLR = learningRate * net.layers{l}.learningRate(j) ;
            
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.layers{l}.weights{j}) ;
                
                if ~run_adam
                    % Update momentum.
                    state.momentum{l}{j} = vl_taccum(...
                        momentum, state.momentum{l}{j}, ...
                        -1, parDer) ;

                    delta = state.momentum{l}{j} ;

                    net_diff_norm = net_diff_norm + norm(thisLR*delta(:))^2;
                    
                    % Update parameters.
                    net.layers{l}.weights{j} = vl_taccum(...
                        1, net.layers{l}.weights{j}, ...
                        thisLR, delta) ;
                    
                else
                    [delta,state.momentum{l}{j},state.variance{l}{j}] = adam_update(t,parDer,state.momentum{l}{j},state.variance{l}{j},thisLR);

                    net_diff_norm = net_diff_norm + norm(delta(:))^2;
                    
                    % Update parameters.
                    net.layers{l}.weights{j} = vl_taccum(...
                        1, net.layers{l}.weights{j}, ...
                        1, delta) ;
                end
            end
        end
    end
end

net_diff_norm = sqrt(net_diff_norm);
end

