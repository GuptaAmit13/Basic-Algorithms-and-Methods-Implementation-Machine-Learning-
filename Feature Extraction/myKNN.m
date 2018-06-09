function [prediction] = myKNN(training_data,test_data,k)
training_data_label = training_data(:,size(training_data,2));
training_data = training_data(:,1:size(training_data,2)-1);
test_data = test_data(:,1:size(test_data,2)-1);
prediction = [];
    for i = 1: size(test_data,1)
        pair_dist =[];
        test_data_row = test_data(i,:);
        %calculating distance for each datapoint from training data
        pair_dist = pdist2(training_data,test_data_row,'euclidean');
        pair_dist = [pair_dist,training_data_label];
        %neareast datapoints first
        pair_dist = sortrows(pair_dist,1);
        %k neareast data points
        pair_dist = pair_dist(1:k,2);
        %output
        prediction = [prediction;mode(pair_dist)];
    end
end
