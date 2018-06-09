
train_data = load('optdigits_train.txt');
val_data = load('optdigits_valid.txt');
val_error = load('validation_error.txt');
Xdatapoint = [];
Ydatapoint = [];
Zdatapoint = [];
value = [];
k=10;
m = [3,6,9,12,15,18];
[a, ind] = min(val_error(:,2));
train_data = [train_data; val_data];
[z,w,v] = mlptrain(train_data,val_data,val_error(ind,1),k);
dlmwrite('w.txt',w);
dlmwrite('v.txt',v);

coef = pca(z);
coef_2 = coef(:,1:2);
coef_3 = coef(:,1:3);
dlmwrite('projection_2.txt',coef_2);
dlmwrite('projection_3.txt',coef_3);
z_pca_2 = z * coef_2;
z_pca_3 = z * coef_3;


%2-D plot%
Xdatapoint = z_pca_2(:,1);
Ydatapoint = z_pca_2(:,2);
value = train_data(:,size(train_data,2));
r = [1:250];
figure;
gscatter(Xdatapoint,Ydatapoint,value);
text(Xdatapoint(r,:),Ydatapoint(r,:),num2str(value(r,:)));


%3-D plot%
Xdatapoint = z_pca_3(:,1);
Ydatapoint = z_pca_3(:,2);
Zdatapoint = z_pca_3(:,3);
value = train_data(:,size(train_data,2));
r = [1:250];
datafile = [Xdatapoint,Ydatapoint,Zdatapoint,value];
color=[0 1 0; 1 0 1; 0 1 1; 1 0 0; .2 .6 1; 1 1 1; 1 .6 .2; 0 0 1; 1 .2 .6; .2 1 .6];
z = datafile(:,1:end-1);
figure
for k=0:9
    z_group= z(datafile(:,end)==k,:);
    scatter3(z_group(:,1),z_group(:,2),z_group(:,3),[],color(k+1,:));
    hold on;
end
rand_array = 1:size(datafile,1);
rand_array = randperm(length(rand_array));
rand_array = rand_array(1:250);
for t=rand_array
    text(z(t,1),z(t,2),z(t,3),num2str(datafile(t,end)));
end
