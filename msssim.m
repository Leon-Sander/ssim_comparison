function [list] = msssim(A, ref)
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


wt = [0.0448 0.2856 0.3001 0.2363 0.1333];
ss = zeros(5, 1);
list = [];
for s = 1 : 5
    ss(s) = ssim(x, y);
    list = [list ss(s)];
    x = impyramid(x, 'reduce');
    y = impyramid(y, 'reduce');
    

end
ms = wt * ss;
list = [list ms];
end