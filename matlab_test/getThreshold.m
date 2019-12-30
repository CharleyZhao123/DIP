function [threshOut]=getThreshold(feature,hist)
[~,C]=kmeans(feature,2);
%thresh = min(C)*0.95+max(C)*0.025+sum(feature)/256*0.025;
countVector = zeros(1,length(feature));
countVector(feature<=min(C)) = 1;
thresh = sum(feature(feature<=min(C)))/(sum(countVector == 1))*0.95 + 0.05*sum(feature(feature>min(C)))/(sum(countVector == 0));
threshOut(1) = -1;
index = 2;
weight = feature(1);
count = 1;
for i = 2:length(feature)
    if((feature(i)>=thresh && feature(i-1)<thresh) || (feature(i-1)>=thresh && feature(i)<thresh))
        threshOut(index) = i;
        weight = 0;
        count = 0;
        index = index + 1;
    end
end
if(mod(length(threshOut)-1,2)==1)
    threshOut = [threshOut,threshOut(length(threshOut))];
end
threshOut(length(threshOut)+1) = 257;
threshOutOrigin = threshOut;
threshOut = zeros(1,length(threshOut)/2);
for i = 1:2:length(threshOutOrigin)
    threshOut(floor(i/2)+1) = (threshOutOrigin(i)+threshOutOrigin(i+1))/2;
end
if(threshOut(1)>0)
    threshOut = [0,threshOut];
end
if(threshOut(length(threshOut))<256)
    threshOut = [threshOut,256];
else
    threshOut(length(threshOut)) = 256;
end
threshOutOrigin = threshOut;
threshOut = zeros(1,2);
index = 1;
left = threshOutOrigin(1);
right = 0;
for i=2:length(threshOutOrigin)
    right = threshOutOrigin(i);
    if(sum(hist(left+1:right))>0)
        threshOut(index,1) = left;
        threshOut(index,2) = right;
        left = right;
        index = index + 1;
    end
end
end