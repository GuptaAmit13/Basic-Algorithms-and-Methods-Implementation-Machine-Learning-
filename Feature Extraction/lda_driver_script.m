train = load('optdigits_train.txt');
test = load('optdigits_test.txt');

train_data = train(:,1:size(train,2)-1);
train_data_label = train(:,size(train,2));
test_data = test(:,1:size(test,2)-1);
test_data_label = test(:,size(test,2));
L = [2,4,9];
k = [1,3,5];    
    for i = 1:3
        [projection,eigenvalues] = myLDA(train,L(i));
        project_train_data = [train_data * projection,train_data_label];
        project_test_data = [test_data*projection,test_data_label];
        error_rate=[];
        for j = 1:3
            [prediction] = myKNN(project_train_data,project_test_data,k(j));
            error_rate = 100 * (size(test_data_label,1) - sum(prediction==test_data_label))/size(test_data_label,1);
            sprintf("L : %d, Error : %0.2f for K : %d",L(i),error_rate,k(j))
        end
    end
