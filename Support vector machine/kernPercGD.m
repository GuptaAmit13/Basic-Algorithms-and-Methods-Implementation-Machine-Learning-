function [alpha, b] = kernPercGD(train_data,train_label)
%train_data = data3;
%train_label = theclass;

k = (train_data*train_data').^2;


n = size(train_data,1);
alpha = zeros(n,1);
b = 0;
delta_err = 100;
pre_err = 0;
iteration = 0;
while abs(delta_err) > 1
    count = 0;
    iteration = iteration + 1;
    for i = 1:n
       summation = k(i,:)*(alpha.*train_label) + b;
       if ( summation * train_label(i,:) <= 0 )
            alpha(i,:) = alpha (i,:) +1;
            b = b + train_label(i,:);
            count = count + 1;
       end
    end
    output_label = k*(alpha.*train_label) + b;
    output_label = sign(output_label);
    err = sum(train_label==output_label);
    delta_err = err-pre_err;
    pre_err = err;
end
fprintf('\n No. of Iteration Required to Converge : %d ',iteration);
end