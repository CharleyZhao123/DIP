function [feature]=enhanceFeature(hist)
feature = zeros(size(hist));
sumHist = sum(hist);
for i = 1:length(hist)
    count = 0;
    sumvalue = 0;
    weight = 0;
    for j = i-10:i+10
        if(j<1 || j>length(hist))
            continue;
        end
        if(hist(j)>(1e-4)*sumHist)
            count = count + (hist(j))*exp(abs(j-i));
        else
            count = count + exp(abs(j-i));
        end
        weight = weight + hist(j);
        if((hist(i)-hist(j))<=0)
            sumvalue = sumvalue + exp((hist(i)-hist(j)))*exp(abs(j-i));
        else
            sumvalue = sumvalue + (hist(i)-hist(j))*exp(abs(j-i));
        end
    end
    if(count>0 && weight>0 && hist(i)>(1e-4)*sumHist)
        feature(i) = sumvalue/count*(1000*hist(i)/weight+hist(i));
    else
        feature(i) = 0;
    end
end
feature = feature/max(feature);
% featureOrigin = feature;
% feature = zeros(size(feature));
% for i = 2:length(feature)
%     feature(i) = featureOrigin(i-1)*0.95 + 0.05*feature(i-1);
% end
end