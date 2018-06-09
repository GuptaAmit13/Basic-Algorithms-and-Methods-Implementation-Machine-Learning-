k = [1,3,5,7];
training_data=load('optdigits_train.txt');
test_data = load('optdigits_test.txt');
for i = 1:4
    [prediction] = myKNN(training_data,test_data,k(i));
    error_rate = 100 * (size(test_data,1) - sum(prediction==test_data(:,size(test_data,2))))/size(test_data,1);
    sprintf("Error : %0.2f for K : %d",error_rate,k(i))
end    

