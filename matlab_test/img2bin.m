function [imgBin]=img2bin(img)
figure;
imshow(img);
img = rgb2gray(img);
hist = imhist(img);
figure;
bar(hist);
[feature]=enhanceFeature(hist);
figure;
plot(1:length(feature),feature);
[threshold]=getThreshold(feature,hist);
display(threshold);
sizeImg = size(img);
imgBin =zeros(sizeImg(1),sizeImg(2),3);
for i=1:sizeImg(1)
    for j = 1:sizeImg(2)
        idx = inThreshold(img(i,j),threshold);
        if(idx == 1)
            imgBin(i,j,:) = [0,0,255];
        elseif(idx == 2)
            imgBin(i,j,:) = [0,255,0];
        elseif(idx == 3)
            imgBin(i,j,:) = [255,0,0];
        elseif(idx == 4)
            imgBin(i,j,:) = [255,255,0];
        elseif(idx == 5)
            imgBin(i,j,:) = [255,0,255];
        elseif(idx == 6)
            imgBin(i,j,:) = [0,255,255];
        elseif(idx == 7)
            imgBin(i,j,:) = [255,255,255];
        end
    end
end
imgBin = uint8(imgBin);
figure;
imshow(imgBin);
end