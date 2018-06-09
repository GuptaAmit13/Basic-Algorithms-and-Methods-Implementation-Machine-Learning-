function [principal_component,eigenvalues] = myPCA(data,num_principal_components)
train_data = data(:,1:size(data,2)-1);
%covariance matrix
train_data_cov = cov(train_data);
[eigenvectors,eigenvalues] = eig(train_data_cov);
%eigen values sorted in descendind order
[eigenvalues,ordering] = sort(diag(eigenvalues),'descend');

eigenvectors = eigenvectors(:,ordering);
PoV = 0;PovGraph=[];
% Assuming if num_principal_components is defined by PoV then it is passed as 0
    if(num_principal_components==0)
        k=0;
        while(PoV < 0.9)
            k=k+1;
            PoV = sum(eigenvalues(1:k))/sum(eigenvalues(:));
            %plotting only till PoV > 0.9 found
            PovGraph = [PovGraph;k,PoV];
        end
    else
        k=num_principal_components;
    end
 if num_principal_components ==0
     %only if num_principal_components is not given.
     plot(PovGraph(:,1),PovGraph(:,2)); hold on;
     scatter(k,PoV,'filled');
 end
 sprintf("num_of_principal_components : %d",k)
 %output
 principal_component = eigenvectors(:,1:k);   
end