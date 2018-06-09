function [mu,sigma,phi] = Myinitialization(data,k)
    [labels, mu] = kmeans(data,k,'EmptyAction','singleton');
    for j = 1:k
        sigma{j} = cov(data(labels==j,:));
        phi(j) = sum(labels==j)/size(data,1);
    end    
end
