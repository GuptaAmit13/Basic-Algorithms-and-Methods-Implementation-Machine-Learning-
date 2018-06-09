function [p1,p2,pc1,pc2] = Bayes_learning(training_data,validation_data)
   
train_data = training_data(:,1:size(training_data,2)-1);
train_data_label = training_data(:,size(training_data,2));

valid_data = validation_data(:,1:size(validation_data,2)-1);
valid_data_label = validation_data(:,size(validation_data,2));

classes = unique(train_data_label);
num_classes = length(classes);
class_i = cell(num_classes,1);
class_prob = cell(num_classes,1);
P_i = cell(num_classes,1);

    %Both Class Calculation   
    for i =1:num_classes
        class_i{i} =train_data(train_data_label==classes(i),:);
        class_prob{i} = sum(class_i{i},1)./size(class_i{i},1);
        P_i{i} = ones(size(class_prob{i}))-class_prob{i};
    end
    
    sigmas = [-5,-4,-3,-2,-1,0,1,2,3,4,5];
    error_rate=[];
    %Using Validation Set for testing sigmas value
    for i=1:size(sigmas,2)
       sigma = sigmas(:,i);
       prior_class_1 = 1/(1+exp(-sigma));
       class_1 = (P_i{1}.^(ones(size(valid_data))-valid_data)).*(class_prob{1}.^valid_data);
       Posterior_class_1 = prior_class_1.*prod(class_1,2);
       prior_class_2 = 1 - prior_class_1;
       class_2 = (P_i{2}.^(ones(size(valid_data))-valid_data)).*(class_prob{2}.^valid_data);
       Posterior_class_2 = prior_class_2.*prod(class_2,2);
       %prediction using current sigma value
       [values, labels] = maxk([Posterior_class_1 Posterior_class_2],1,2);
        diff = labels - valid_data_label;
        %error for current sigma value
        error_rate =[error_rate;100*(1-(size(labels,1) - nnz(diff))/size(labels,1))];
        sprintf("Error: %0.2f for sigma %d :",error_rate(i),sigma)
    end
   %Selecting best sigma
    sprintf("Sigma : Error Rate")
    [sigmas',error_rate]
    [values,index] = min(error_rate);
    final_sigma = sigmas(index);
    sprintf("Best Sigma : %d ",final_sigma)
    %parameter of bernoulli
    p = cell2mat(P_i);
    p1 = p(1,:);
    p2= p(2,:);
    pc1 = 1/(1+exp(-final_sigma));
    pc2 = 1-pc1;
end
