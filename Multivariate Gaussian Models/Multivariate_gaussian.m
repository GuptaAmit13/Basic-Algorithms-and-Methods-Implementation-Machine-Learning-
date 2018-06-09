% Separating Training Data
complete_matrix=table2array(readtable('training_data.txt'));
data = [];
label = [];
X1_data=[];
X1_label=[];
X2_data=[];
X2_label=[];
for(i = 1:100)
    data = [data;complete_matrix(i,1:8)];
    label=[label;complete_matrix(i,9)];
    if(complete_matrix(i,9)==1.0000)
        X1_data=[X1_data;complete_matrix(i,1:8)];
        X1_label=[X1_label;complete_matrix(i,9)];
    else
        X2_data=[X2_data;complete_matrix(i,1:8)];
        X2_label=[X2_label;complete_matrix(i,9)];
    end
end

% Separating Test Data
complete_test_matrix = table2array(readtable('test_data.txt'));
test_data = [complete_test_matrix(:,1:8)];
test_label = [complete_test_matrix(:,9)];


%---------------------------------------------------------------------------------------------------------------------------------%
% Model 1 
S1 = cov(X1_data);
S2 = cov(X2_data);
M1 = mean(X1_data);
M2 = mean(X2_data);
c1 =0.6;
c2 =0.4;

G1_data =[];
G2_data =[];
%Calculating Model
for (i = 1:size(test_data))
    G1_data = [G1_data;-(1/2)*log(det(S1)) - (1/2) *(test_data(i,:) * inv(S1) * (test_data(i,:))' - 2 * (test_data(i,:)) * inv(S1)* M1' + M1*inv(S1)*M1')  + log(c1)];
end
for (i = 1:size(test_data))
    G2_data = [G2_data;-(1/2)*log(det(S2)) - (1/2) *(test_data(i,:) * inv(S2) * (test_data(i,:))' - 2 * (test_data(i,:)) * inv(S2)* M2' + M2*inv(S2)*M2')  + log(c2)];
end

% Assigning Classes to data Points
result_matrix(G1_data>= G2_data)=1;
result_matrix(G1_data< G2_data)=2;

%Co-variance Used
S1
S2
%Checking the Error Rate of Model 1
Error_Rate_percentage = (100-sum(test_label==result_matrix'))




%---------------------------------------------------------------------------------------------------------------------------------%
% Model 2
S1 = cov(X1_data);
S2 = cov(X2_data);
M1 = mean(X1_data);
M2 = mean(X2_data);
c1 =0.6;
c2 =0.4;
%Shared Co-variance
S = S1*c1 + S2*c2;
G1_data =[];
G2_data =[];
%Calculating Model
for (i = 1:size(test_data))
    G1_data = [G1_data; - (1/2) *(test_data(i,:) * inv(S) * (test_data(i,:))' - 2 * (test_data(i,:)) * inv(S)* M1' + M1*inv(S)*M1')  + log(c1)];
end
for (i = 1:size(test_data))
    G2_data = [G2_data; - (1/2) *(test_data(i,:) * inv(S) * (test_data(i,:))' - 2 * (test_data(i,:)) * inv(S)* M2' + M2*inv(S)*M2')  + log(c2)];
end

%Assigning Class to Data Point
result_matrix(G1_data>= G2_data)=1;
result_matrix(G1_data< G2_data)=2;

%Co-variance Used
S1
S2
%Shared Covariance 
S
%Error Rate of Model 2 
Error_Rate_percentage = (100-sum(test_label==result_matrix'))


%---------------------------------------------------------------------------------------------------------------------------------%

% Model 3
S1 = cov(X1_data);
% Diagonal of Covariance  
S1_diag = diag(S1);
S2 = cov(X2_data);
% Diagonal of Covariance
S2_diag = diag(S2);
M1 = mean(X1_data);
M2 = mean(X2_data);
c1 =0.6;
c2 =0.4;
G1_data =[];
G2_data =[];
summation=0;
% doing prod isntead of det because of diagonal matrix
for (i = 1:size(test_data))
    for(j = 1:size(test_data,2))
        summation = summation + ((test_data(i,j)-M1(j))/S1_diag(j)).^2;
    end
    G1_data =[G1_data; - (1/2) *log(prod(S1_diag))-(1/2) * (summation)+ log(c1)];
    summation=0;
end
%doing prod instead of det because of diagonal matrix
for (i = 1:size(test_data))
    for(j = 1:size(test_data,2))
        summation = summation + ((test_data(i,j)-M2(j))/S2_diag(j)).^2;
    end
    G2_data =[G2_data; - (1/2) *log(prod(S2_diag))-(1/2) * (summation)+ log(c1)];
    summation=0;
end

%Assigning Class to Data points
result_matrix(G1_data>= G2_data)=1;
result_matrix(G1_data< G2_data)=2;

%Diagonal Covariance
S1_diag
%Diagonal Covariance
S2_diag

%Error Rate Model 3
Error_Rate_percentage = (100-sum(test_label==result_matrix'))


%---------------------------------------------------------------------------------------------------------------------------------%

% Model 4
S1 = cov(X1_data);
S1_diag = diag(S1);
%Equal Diagonal variance
S1_equal_var = mean(S1_diag);
S2 = cov(X2_data);
S2_diag = diag(S2);
%Equal Diagonal variance
S2_equal_var = mean(S2_diag);
M1 = mean(X1_data);
M2 = mean(X2_data);
c1 =0.6;
c2 =0.4;
G1_data =[];
G2_data =[];
summation=0;
% doing prod isntead of det because of diagonal matrix
for (i = 1:size(test_data))
    for(j = 1:size(test_data,2))
        summation = summation + ((test_data(i,j)-M1(j))/S1_equal_var).^2;
    end
    G1_data =[G1_data; - (1/2) *log(prod(S1_equal_var))-(1/2) * (summation)+ log(c1)];
    summation=0;
end
%doing prod instead of det because of diagonal matrix
for (i = 1:size(test_data))
    for(j = 1:size(test_data,2))
        summation = summation + ((test_data(i,j)-M2(j))/S2_equal_var).^2;
    end
    G2_data =[G2_data; - (1/2) *log(prod(S2_equal_var))-(1/2) * (summation)+ log(c1)];
    summation=0;
end

%Assigning Class to Data Points
result_matrix(G1_data>= G2_data)=1;
result_matrix(G1_data< G2_data)=2;

%Covariance of Diagonal
S1_equal_var
%Covariance of Diagonal
S2_equal_var
%Error Rate of Model 4
Error_Rate_percentage = (100-sum(test_label==result_matrix'))


%---------------------------------------------------------------------------------------------------------------------------------%
