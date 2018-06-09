function [projection,eigenvalues] = myLDA(data,num_principal_components)

train_data = data(:,1:size(data,2)-1);
%complete data mean
train_data_mean = mean(train_data);
train_data_label = data(:,size(data,2));
classes = unique(train_data_label);
k = length(classes);
Class_data = cell(k,1);
Class_data_mean = cell(k,1);
S_within_class = cell(k,1);
Sw=0;
Sb=0;
    for j= 1:k
        Class_data{j} = train_data(train_data_label==classes(j),:);
        N(j) = size(Class_data{j},1);
        Class_data_mean{j} = mean(Class_data{j});
        %within class calculations
        S_within_class{j} = 0;
        for i = 1:size(Class_data{j},1)
            S_within_class{j} = S_within_class{j} + (Class_data{j}(i,:) - Class_data_mean{j})'*(Class_data{j}(i,:)-Class_data_mean{j});
        end
       Sw = Sw + S_within_class{j};
       %between class calculation
        Sb = Sb + N(j)*(Class_data_mean{j}-train_data_mean)'*(Class_data_mean{j}-train_data_mean);
    end
% inverse of sw, in this psuedo inverse since inverse is not possible    
Solution = pinv(Sw) * Sb;
[W,lambda] = eig(Solution);
[lambda,eigenvalues] = sort(diag(lambda),'descend');
W = W(:,eigenvalues);
%output
projection = W(:,1:num_principal_components);
end