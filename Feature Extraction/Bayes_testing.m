function [error_rate] = Bayes_testing(test,p1,p2,pc1,pc2)
test_data = test(:,1:size(test,2)-1);
test_data_label = test(:,size(test,2));

        %Class 1 Prob Calculations 
        class_prob_1= ones(size(p1))-p1;
        %likelihood
        class_1=(p1.^(ones(size(test_data))-test_data)).*(class_prob_1.^test_data);
        %Posterior
        posterior_class_1=pc1.*prod(class_1,2);
        
        %Class 2 Prob Calculations
        class_prob_2= ones(size(p2))-p2;
        %likelhood
        class_2=(p2.^(ones(size(test_data))-test_data)).*(class_prob_2.^test_data);
        %Posterior
        posterior_class_2=pc2.*prod(class_2,2);
    
     %prediction class for datapoints
    [values, prediction] = maxk([posterior_class_1,posterior_class_2],1,2);
    %calculating difference
     diff = prediction - test_data_label;
     %error
     error_rate =100*(1-(size(prediction,1) - nnz(diff))/size(prediction,1));   
end