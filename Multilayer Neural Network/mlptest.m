function [z] = mlptest(test_data,w,v)
test_label = test_data(:,size(test_data,2));
n = size(test_data,1);
d = size(test_data,2)-1;
X = test_data(:,1:d);
X = [ones(n,1),X];
result = zeros(n,1);
m = size(w,1);
k=10;
for selected_row = 1:n
    for h = 1:m
            res = X(selected_row,:) * w(h,:)';
            if(res < 0 )
                z(selected_row,h+1) = 0;
            else
                z(selected_row,h+1) = res;
            end
    end
    output(selected_row,:) = z(selected_row,:)*v';
    for k = 1:k
        y(selected_row,k) = exp(output(selected_row,k))/sum(exp(output(selected_row,:)));
    end
        [a, index] = max(y(selected_row,:));
        result(selected_row) = index-1;
end

fprintf('\nHidden Units: %d \tError test : %f \n',m,100 - 100*sum(test_label==result)/n);
end