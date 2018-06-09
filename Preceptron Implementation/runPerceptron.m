load('data1.mat');
w0 = [1;-1] ; % intial w Vector 

[w,step] = MyPerceptron(X,y,w0);

