
%Q1
k=[4,8,12];
flag =1;i=1;
while flag ==1
    try
        [H,M,Q] = EMG(0,'stadium.bmp',k(i));
        img = reshape(H,67,200,3);
        figure();
        imagesc(img);
        hold on;
        i=i+1;
        if(i>3)
            flag=0;
        end
    catch 
        warning("Singular Sigma Matrix created !!")
        warning("Restarting EM !!")
        flag=1;
    end     
end

%Q3

    try
        [H,M,Q] = EMG(0,'goldy.bmp',7);
        flag=false;
    catch 
        warning("Singular Matrix created !!")
        warning(" EM Failure!!");
    end
         
 [img cmap] = imread('goldy.bmp');
 img_rgb = ind2rgb(img,cmap);
 img_double = im2double(img_rgb);
 data = reshape(img_double,[],3);
 [labels, mu] = kmeans(data,7);
 for i=1:size(data,1)
    data(i,:) = mu(labels(i),:);
 end
 img = reshape(data,115,150,3);
 figure();
 imagesc(img);
 hold on;
 
 %Q4
    try
        [H,M,Q] = EMG(1,'goldy.bmp',7);
        img = reshape(H,115,150,3);
        figure();
        imagesc(img);
        hold on;
    catch 
        warning("Singular Matrix created !!")
        warning(" EM Failure!!");
    end
 