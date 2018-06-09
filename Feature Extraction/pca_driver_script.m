data= load('optdigits_train.txt');
test = load('optdigits_test.txt');
num_principal_components = 0;
[principal_components,eigenvalues] = myPCA(data,num_principal_components);
train_data = data(:,1:size(data,2)-1)- mean(data(:,1:size(data,2)-1));
train_data_label= data(:,size(data,2));
test_data = test(:,1:size(test,2)-1)-mean(test(:,1:size(test,2)-1));
test_data_label = test(:,size(test,2));
projected_train_data = [train_data * principal_components,train_data_label];
projected_test_data = [test_data * principal_components,test_data_label];
k=[1,3,5,7];
for i = 1:4
    [prediction] = myKNN(projected_train_data,projected_test_data,k(i));
    error_rate = 100 * (size(test_data_label,1) - sum(prediction==test_data_label))/size(test_data_label,1);
    sprintf("Error : %0.2f for K : %d",error_rate,k(i))
end

