addpath(genpath(cd))
clear

pic_name = './test5.jpg';
X = double(imread(pic_name));
X = X(41:40+256, 51:50+256, :);

X = X/255;
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
Xn = X;
rhos = 0.1
ind = find(rand(n1*n2*n3,1)<rhos);
Xn(ind) = rand(length(ind),1);

opts.mu = 1e-4;
opts.tol = 1e-5;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 1;

[n1,n2,n3] = size(Xn);
lambda = 1/sqrt(max(n1,n2)*n3);
[Xhat,E,err,iter] = trpca_tnn(Xn,lambda,opts);
 
Xhat = max(Xhat,0);
Xhat = min(Xhat,maxP);
psnr = PSNR(X,Xhat,maxP)

%% inpainting RobustPCA example: moon picture corrupted with some text
addpath('../');

% read image and add the mask
[img] = imread(pic_name);
img = img(41:40+256, 51:50+256, :);
%[indImg, map] = rgb2ind(img, 256);
%figure(1)
%imshow(indImg, map)
%[img, map] = imread('moon.tif');
%img = rgb2gray(img);
%newmap = rgb2gray(map);
img = double(img) / 255;

%%msk = zeros(size(img));
%msk(65:192,65:192) = imresize(imread('text.png'), 0.5);
%img_corrupted = img;
%img_corrupted(msk > 0) = nan;

[n1,n2,n3] = size(img);
img_corrupted = img;
img_na = img;
rhos = 0.1;
ind = find(rand(n1*n2*n3,1)<rhos);
img_na(ind) = nan;
img_corrupted(ind) = rand(length(ind),1);

% create a matrix X from overlapping patches
ws = 8; % window size
no_patches = size(img, 1) / ws;
X = zeros(no_patches^2, ws^2);
k = 1;
for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        patch = img_corrupted(r1, r2);
        X(k,:) = patch(:);
        k = k + 1;
    end
end

% apply Robust PCA
lambda = 0.02; % close to the default one, but works better
tic
[L, S] = RobustPCA(X, lambda, 1.0, 1e-5);
toc

% reconstruct the image from the overlapping patches in matrix L
img_reconstructed = zeros(size(img));
img_noise = zeros(size(img));
k = 1;
for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        % average patches to get the image back from L and S
        % todo: in the borders less than 4 patches are averaged
        patch = reshape(L(k,:), ws, ws);
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        img_reconstructed(r1, r2, 1) = img_reconstructed(r1, r2, 1) + 0.25*patch;
        img_reconstructed(r1, r2, 2) = img_reconstructed(r1, r2, 2) + 0.25*patch;
        img_reconstructed(r1, r2, 3) = img_reconstructed(r1, r2, 3) + 0.25*patch;
        patch = reshape(S(k,:), ws, ws);
        img_noise(r1, r2, 1) = img_noise(r1, r2, 1) + 0.25*patch;
        img_noise(r1, r2, 2) = img_noise(r1, r2, 2) + 0.25*patch;
        img_noise(r1, r2, 3) = img_noise(r1, r2, 3) + 0.25*patch;
        k = k + 1;
    end
end
img_final = img_reconstructed;
img_final(~isnan(img_na)) = img_corrupted(~isnan(img_na));

L = max(L,0);
L = min(L,maxP);
psnr = PSNR(X,L,maxP)

% show the results
%img_corrupted = uint8(img_corrupted * 255);
%size(img_corrupted)
%img_corrupted_rgb = ind2rgb(uint8(img_corrupted * 255), map);
%img_final_rgb = ind2rgb(uint8(img_final * 255), map);

%figure(1)
%subplot(1,4,1)
imwrite(img, "./PaperResult/test5_10_orig.png")
%subplot(1,4,2)
imwrite(Xn/max(Xn(:)), "./PaperResult/test5_10_corr.png")
%subplot(1,4,3)
imwrite(img_final, "./PaperResult/test5_10_tprca.png")
%subplot(1,4,4)
imwrite(Xhat/max(Xhat(:)), "./PaperResult/test5_10_prca.png")
