train = load('face_train_data_960.txt');
test =  load('face_test_data_960.txt');
data = [train;test];
[principal_components,eigenvalues] = myPCA(data,5);
principal_components = principal_components';
for i = 1:5
    subplot(1,5,i);
    imagesc(reshape(principal_components(i,1:size(principal_components,2)),32,30)'); hold on;
end