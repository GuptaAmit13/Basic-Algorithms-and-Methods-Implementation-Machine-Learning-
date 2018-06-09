function [z,w,v] = mlptrain(train_data,val_data,m,k)

learning_rate = 0.0001;

train_label = train_data(:,size(train_data,2))+1;
label = zeros(numel(train_label), max(train_label));
label(sub2ind(size(label), 1:numel(train_label), train_label')) = 1;
n = size(train_data,1);
d = size(train_data,2)-1;

X = train_data(:,1:d);
X = [ones(n,1),X];

w = zeros(m,d+1);
z = ones(n,m+1);
v = zeros(k,m+1); 
e= [];
w = -0.01 + (0.01+0.01)*rand(m,d+1);
v = -0.01 + (0.01+0.01)*rand(k,m+1);
  

delta_v = zeros(k,m+1);
delta_w = zeros(m,d+1);

output = zeros(n,k);
y = output;
flag =true;
while flag == true 
    rand_array = 1:size(X,1);
    rand_array = randperm(length(rand_array));
    for selected_row = rand_array
        
        for h = 1:m
            res = X(selected_row,:) * w(h,:)';
            if(res < 0 )
                z(selected_row,h+1) = 0;
            else
                z(selected_row,h+1) = res;
            end
        end
        
        output(selected_row,:) = z(selected_row,:)*v';
        for k = 1:k
            y(selected_row,k) = exp(output(selected_row,k))/sum(exp(output(selected_row,:)));
        end
        
        delta_v = learning_rate*((label(selected_row,:)-y(selected_row,:))'*z(selected_row,:));
        
        for h = 1:m
            res = X(selected_row,:) * w(h,:)';
            if(res < 0 )
                delta_w(h,:) = 0;
            else
                delta_w(h,:) = learning_rate*(((label(selected_row,:)-y(selected_row,:))*v(:,h+1))'*X(selected_row,:));
            end
        end
        
        v = v + delta_v;
        w = w + delta_w;      
    end
    e = [e,sum(sum(label.*log(y)))];
    
    if ( e(size(e,2)) < mean(e(1:size(e,2)-1)))
        delta_learning_rate = 0.00001;
    else
        delta_learning_rate = - 0.001 * learning_rate; 
    end
    learning_rate = learning_rate + delta_learning_rate;
    
    if(size(e,2)>1)
        if( (e(size(e,2)) - e(size(e,2)-1)) < 10e-4 )
            flag =false;
        end
    end
end


%Training Data


n = size(train_data,1);
d = size(train_data,2)-1;
result = zeros(n,1);
train_label = train_label -1;
for selected_row = 1:n
    for h = 1:m
            res = X(selected_row,:) * w(h,:)';
            if(res < 0 )
                z(selected_row,h+1) = 0;
            else
                z(selected_row,h+1) = res;
            end
    end
    output(selected_row,:) = z(selected_row,:)*v';
    for k = 1:k
        y(selected_row,k) = exp(output(selected_row,k))/sum(exp(output(selected_row,:)));
    end
        [a, index] = max(y(selected_row,:));
        result(selected_row) = index-1;
end

fprintf('\nHidden Units: %d \tError Training : %f \n',m,100 - 100*sum(train_label==result)/n);
if(m==3)
    fileID = fopen('train_error.txt','w');
    res = 100 - 100*sum(train_label==result)/n;
    fprintf(fileID,'%d %f\n',m,res);
    fclose(fileID);
else
    fileID = fopen('train_error.txt','a');
    res = 100 - 100*sum(train_label==result)/n;
    fprintf(fileID,'%d %f\n',m,res);
    fclose(fileID);
end


%Validation Data
val_label = val_data(:,size(val_data,2));
n = size(val_data,1);
d = size(val_data,2)-1;
X = val_data(:,1:d);
X = [ones(n,1),X];
result = zeros(n,1);

for selected_row = 1:n
    for h = 1:m
            res = X(selected_row,:) * w(h,:)';
            if(res < 0 )
                z(selected_row,h+1) = 0;
            else
                z(selected_row,h+1) = res;
            end
    end
    output(selected_row,:) = z(selected_row,:)*v';
    for k = 1:k
        y(selected_row,k) = exp(output(selected_row,k))/sum(exp(output(selected_row,:)));
    end
        [a, index] = max(y(selected_row,:));
        result(selected_row) = index-1;
end

fprintf('Hidden Units: %d \tError Validation : %f \n',m,100 - 100*sum(val_label==result)/n);
if(m==3)
    fileID = fopen('validation_error.txt','w');
    res = 100 - 100*sum(val_label==result)/n;
    fprintf(fileID,'%d %f\n',m,res);
    fclose(fileID);
else
    fileID = fopen('validation_error.txt','a');
    res = 100 - 100*sum(val_label==result)/n;
    fprintf(fileID,'%d %f\n',m,res);
    fclose(fileID);
end
end