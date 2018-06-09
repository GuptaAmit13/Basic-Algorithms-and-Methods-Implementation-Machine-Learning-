function [H,M,Q] = EMG(flag,image,k)
    [img cmap] = imread(image);
    img_rgb = ind2rgb(img,cmap);
    img_double = im2double(img_rgb);
    data = reshape(img_double,[],3);
    
    %Initialization Step
    [mu,sigma,phi] = Myinitialization(data,k);

    %Learning Step
    [mu_final,sigma_final,lamda,Q] = Mylearning(flag,data,mu,sigma,phi,k);

    %Assigning Step
    [values index] = max(lamda,[],2);
    for i=1:size(data,1)
        data(i,:) = mu_final(index(i),:);
    end    
    H = data;
    M = mu_final
end
