function [list,  x, y] = msssim(A, ref)
%EVALUNWARP compute MSSSIM between images
%   A:      image1
%   ref:    reference image
%   list:   returnes 5 ssim values and the msssim value
%   Matlab image processing toolbox is necessary to compute ssim. The weights 
%   for multi-scale ssim is directly adopted from:
%
%   Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural 
%   similarity for image quality assessment." In Signals, Systems and Computers, 
%   2004. Conference Record of the Thirty-Seventh Asilomar Conference on, 2003. 

x = imread(A);
y = imread(ref);
size(x)
size(y)
x = rgb2gray(x);
y = rgb2gray(y);
size(x)
%image(x)
wt = [0.0448 0.2856 0.3001 0.2363 0.1333];
ss = zeros(5, 1);
list = [];
x_reduced = 0;
y_reduced = 0;
for s = 1 : 5
    size(x_reduced)
    if s == 1
        ss(s) = ssim(x, y);
        list = [list ss(s)];
        x_reduced = impyramid(x, 'reduce');
        y_reduced = impyramid(y, 'reduce');
    else
        ss(s) = ssim(x_reduced, y_reduced);
        list = [list ss(s)];
        x_reduced = impyramid(x_reduced, 'reduce');
        y_reduced = impyramid(y_reduced, 'reduce');
    end

end
ms = wt * ss;
list = [list ms];
end