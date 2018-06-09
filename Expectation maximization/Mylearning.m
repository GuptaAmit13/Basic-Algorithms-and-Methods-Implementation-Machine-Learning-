function [mu_final,sigma_final,lamda,Q] = Mylearning(flag,data,mu,sigma,phi,k)
    Q_e = [];
    Q_m = [];
    Q=[];
    logliklihoodThresNew =0;
    logliklihoodThresOld=0;
    logliklihoodThreshold = 10e-5;
    lambda=0.0001;
    
    %to hold Probability Density for each point for each cluster for each point.
     pdf = zeros(size(data,1),k);
            
    %Until converge or 150 iterations. whichever comes first.
    for iter = 1:120
         %Weighted matrix 
        W = zeros(size(data,1),k);
        log_likelihood_e=0;
        log_likelihood_m=0;
        
        %Expectation Step
        
            for j = 1:k
                    pdf(:,j) = mvnpdf(data,mu(j,:),sigma{j});
            end
            %Multiplying pdf with weight
            pdf_w = bsxfun(@times,pdf,phi);

            %Dividing the weighted prob by the sum of weighted prob for each
            %cluster
            W = bsxfun(@rdivide,pdf_w,sum(pdf_w,2));
            
            for i = 1:size(data,1)
                for j = 1:k
                        log_likelihood_e = log_likelihood_e + W(i,j)*(log(phi(j)+eps)+log(eps + pdf(i,j)));
                end 
            end
            Q_e = [Q_e;log_likelihood_e];
            
        %Maximization Step
            [value,cluster]= max(W,[],2);
           
            for j=1:k
                sigma_k = zeros(size(data,2),size(data,2));
                N_cluster = sum(W(:,j));
                phi(j) = N_cluster/size(cluster,1);
                mu(j,:) = sum(W(:,j).*data)/N_cluster;
                cluster_mean = bsxfun(@minus,data,mu(j,:));
                for i=1:size(data,1)
                    sigma_k = sigma_k + (W(i,j)*(cluster_mean(i,:)'*cluster_mean(i,:)));
                end
                sigma{j} = bsxfun(@rdivide,(sigma_k + flag*lambda*eye(size(data,2))),N_cluster);
            end
            for j = 1:k
                    pdf(:,j) = mvnpdf(data,mu(j,:),sigma{j});             
            end
            
            for i = 1:size(data,1)
                for j = 1:k
                    log_likelihood_m = log_likelihood_m + W(i,j)*(log(phi(j)+eps) + log(eps + pdf(i,j)));
                end
            end  
            logliklihoodThresNew = log_likelihood_m;
          Q_m= [Q_m;log_likelihood_m];
          val = abs((logliklihoodThresNew - logliklihoodThresOld)/logliklihoodThresOld);
         if iter==150 || abs(val)<logliklihoodThreshold
            mu_final = mu;
            sigma_final = sigma;
            lamda = W;
            break;
         end
         logliklihoodThresOld = logliklihoodThresNew;
         Q = [Q;Q_e;Q_m];
    end
    figure();
    xlabel('Iteration Number');
    ylabel('Expected Complete Log Likelihood');
    scatter(1:size(Q_e,1), Q_e,'ro');
    hold on;
    scatter(1:size(Q_m,1), Q_m,'bx');
    legend('After Expectation','After Maximization');
 
end
