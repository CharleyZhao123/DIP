function [out]=inThreshold(gray,threshold)
sizeThreshold = size(threshold);
out = 0;
for i=1:sizeThreshold(1)
    if(gray>=threshold(i,1) && gray<threshold(i,2))
        out = i;
        break;
    end
end
end