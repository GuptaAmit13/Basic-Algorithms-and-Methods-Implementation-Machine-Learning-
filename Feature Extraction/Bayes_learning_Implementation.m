train =load('SPECT_train.txt');
test = load('SPECT_test.txt');
valid = load('SPECT_valid.txt');
[p1,p2,pc1,pc2] = Bayes_learning(train,valid);
error = Bayes_testing(test, p1, p2, pc1, pc2);
sprintf("error on test set : %0.2f",error)