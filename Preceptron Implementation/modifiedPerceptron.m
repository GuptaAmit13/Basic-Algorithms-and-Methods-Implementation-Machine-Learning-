load('data2.mat');
[m,n]=size(X);
f=[zeros(n,1);ones(m,1)]; % transform problem into a standard LP
A1=[X.*repmat(y,1,n),eye(m,m)];
A2=[zeros(m,n),eye(m,m)];
A=-[A1;A2];
b=[-ones(m,1);zeros(m,1)];
x = linprog(f,A,b);% solve LP
w=x(1:n);% return varible w
sz= 15;
a = X(:,1);
b = -w(1)*a/w(2);
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
