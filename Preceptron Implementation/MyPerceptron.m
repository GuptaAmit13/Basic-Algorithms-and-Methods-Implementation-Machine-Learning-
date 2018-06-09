function [w,step] = MyPerceptron(X,y,w0)
%Linear Classifier

%%Visual Part Code
sz= 15;
a = X(:,1);
b = -w0(1)*a/w0(2);
hold on
for i = 1:size(y)
    if(y(i)==1)
        scatter(X(i,1),X(i,2),sz,'b','filled')
    else
        scatter(X(i,1),X(i,2),sz,'r','filled')
    end
end
plot(a,b,'k');
axis([-1.5 1.5 -1.5 1.5]);
hold off

%%Main Function
N = size(X,1);
err = 1;
count=0;
while err >0
    err_count=0;
    for row_no = 1: N
        if sign(X(row_no,:)*w0) ~= y(row_no)
            w0=w0 + 1 * X(row_no,:)'*y(row_no); % 1 rate
            err_count=err_count+1;
        end
    end
    count=count+1;
    err = err_count/N;
end
w = w0;
step = count;

%%After Algorithm visual part
b = -w(1)*a/w(2);
figure 
hold on
for i = 1:size(y)
    if(y(i)==1)
        scatter(X(i,1),X(i,2),sz,'b','filled')
    else
        scatter(X(i,1),X(i,2),sz,'r','filled')
    end
end
plot(a,b,'k');
axis([-1.5 1.5 -1.5 1.5]);

end
