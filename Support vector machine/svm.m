%Question 2

%% Data Generation
rng(1); % For reproducibility
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1); % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
r2 = sqrt(3*rand(100,1)+2); % Radius
t2 = 2*pi*rand(100,1); % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

%Data Visualization
h = [];
figure;
h(1)=plot(data1(:,1),data1(:,2),'r.','MarkerSize',15);
hold on
h(2)= plot(data2(:,1),data2(:,2),'b.','MarkerSize',15);
h(3)=ezpolar(@(x)1);ezpolar(@(x)2);
legend(h,{'-1','+1','Decision Boundary'})
axis equal
hold off

%Data Aggregation
data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;


%Question 2 Subpart A
n = size(data3,1);
k = (data3*data3').^2;
[alpha , b] = kernPercGD(data3,theclass);
output_label = k*(alpha.*theclass) + b;
output_label = sign(output_label);
error = (100 - 100*sum(theclass==output_label)/n);
fprintf('\n Question 2 Subpart A Error Rate on Training data = %d \t', error);
%Plotting 
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

k = (xGrid*(data3.')).^2;
%n = size(xGrid, 1);
mesh_y = k*(alpha.*theclass) + b;

figure;
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
contour(x1Grid,x2Grid,reshape(mesh_y,size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Decision Boundary'})
axis equal
hold off

%% Question 2 Subpart B
%Train the SVM Classifier
h = [];
%BoxConstraint 0.0001
cl = fitcsvm(data3,theclass,'KernelFunction','polynomial','BoxConstraint',1e-4,'ClassNames',[-1,1]);

%BoxConstraint 1
c2 =fitcsvm(data3,theclass,'KernelFunction','polynomial','BoxConstraint',1,'ClassNames',[-1,1]);

%Box Constraint Inf
c3 = fitcsvm(data3,theclass,'KernelFunction','polynomial','BoxConstraint',Inf,'ClassNames',[-1,1]);

% Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);
[~,scores1] = predict(c2,xGrid);
[~,scoresInf] = predict(c3,xGrid);

% Plot the data and the decision boundary
figure;
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(0,0,'co');
h(4) = plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),'ko');
h(5) = plot(data3(c2.IsSupportVector,1),data3(c2.IsSupportVector,2),'go');
h(6) = plot(data3(c3.IsSupportVector,1),data3(c3.IsSupportVector,2),'yo');

contour(x1Grid,x2Grid,reshape(mesh_y,size(x1Grid)),[0 0],'c');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'g');
contour(x1Grid,x2Grid,reshape(scoresInf(:,2),size(x1Grid)),[0 0],'y');
legend(h,{'-1','+1','My Kernel Boundary','fitcsvm Boundary, BoxConstraint 0.0001','Support Vectors,BoxConstraint 1','Support Vectors,BoxConstraint Inf'});
axis equal
hold off


%% Question 2 Subpart C

%Digit 4 & 9 
data = load('optdigits49_train.txt');
train_data = data(:,1:size(data,2)-1);
train_label = data(:,size(data,2));
data = load('optdigits49_test.txt');
test_data = data(:,1:size(data,2)-1);
test_label = data(:,size(data,2));

n = size(train_data,1);

[alpha , b] = kernPercGD(train_data,train_label);
%Train Data
k = (train_data*train_data').^2;
output_label = k*(alpha.*train_label) + b;
output_label = sign(output_label);

error = (100 - 100*sum(train_label==output_label)/n);
fprintf('\n Question 2 Subpart C Error Rate on Training data (4,9) = %f \t', error);

%Test Data
n = size(test_data,1);
k = (test_data*train_data').^2;
output_label = k*(alpha.*train_label) + b;
output_label = sign(output_label);

error = (100 - 100*sum(test_label==output_label)/n);
fprintf('\n Question 2 Subpart c Error Rate on Test data (4,9) = %f \t', error);

% Digit 7 & 9
data = load('optdigits79_train.txt');
train_data = data(:,1:size(data,2)-1);
train_label = data(:,size(data,2));
data = load('optdigits79_test.txt');
test_data = data(:,1:size(data,2)-1);
test_label = data(:,size(data,2));

n = size(train_data,1);

[alpha , b] = kernPercGD(train_data,train_label);
%Train Data
k = (train_data*train_data').^2;
output_label = k*(alpha.*train_label) + b;
output_label = sign(output_label);

error = (100 - 100*sum(train_label==output_label)/n);
fprintf('\n Question 2 Subpart C Error Rate on Training data (7,9) = %f \t', error);

%Test Data
n = size(test_data,1);
k = (test_data*train_data').^2;
output_label = k*(alpha.*train_label) + b;
output_label = sign(output_label);

error = (100 - 100*sum(test_label==output_label)/n);
fprintf('\n Question 2 Subpart C Error Rate on Test data (7,9) = %f \t', error);
