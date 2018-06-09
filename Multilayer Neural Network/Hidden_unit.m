train_data = load('optdigits_train.txt');
val_data = load('optdigits_valid.txt');
test_data = load ('optdigits_test.txt');
m = [3,6,9,12,15,18];
k=10;
for i = m   
    [z,w,v] = mlptrain(train_data,val_data,i,k);
end


val_error = load('validation_error.txt');
val_error = val_error(1:end-2,:);
[a, ind] = min(val_error(:,2));
fprintf('\nThe Best Number of Hidden Units is : %d',val_error(ind,1));
train_error = load('train_error.txt');
train_error = train_error(1:end-2,:);

figure();
plot(m,train_error(:,2),'r',m,val_error(:,2),'b');
legend('Train Error','Validation Error')
[z,w,v] = mlptrain(train_data,val_data,val_error(ind,1),k);
[z] = mlptest(test_data,w,v);
