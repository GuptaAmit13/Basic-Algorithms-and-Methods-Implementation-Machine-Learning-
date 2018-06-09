data= load('face_train_data_960.txt');
num_principal_components = [10,50,100];
plotno=1;
    for i = 1:3
        [principal_components,eigenvalues] = myPCA(data,num_principal_components(i));
        train_data = data(:,1:size(data,2)-1)- mean(data(:,1:size(data,2)-1));
        train_data_label= data(:,size(data,2));
        projected_train_data = train_data * principal_components;
        back_projected_data = projected_train_data * principal_components' + mean(data(:,1:size(data,2)-1));
        for j = 1:5
            subplot(3,5,plotno);
            imagesc(reshape(back_projected_data(j,1:size(back_projected_data,2)),32,30)'); hold on;
            plotno=plotno+1;
        end
    end
    
