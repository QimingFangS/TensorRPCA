%% video RobustPCA example: separates background and foreground
addpath('../');

% ! the movie will be downloaded from the internet !
movieFile = '1292.mp4';
%urlwrite('https://github.com/QimingFangS/Downloads/blob/main/test.mp4?raw=true', movieFile);

% open the movie
n_frames = 100;
movie = VideoReader(movieFile);
frate = movie.FrameRate;    
height = movie.Height;
width = movie.Width;

% vectorize every frame to form matrix X
X = zeros(n_frames, height*width, 3);
for i = (1:n_frames)
    frame = read(movie, i);
    %frame = rgb2gray(frame);
    X(i,:,1) = reshape(frame(:,:,1),[],1);
    X(i,:,2) = reshape(frame(:,:,2),[],1);
    X(i,:,3) = reshape(frame(:,:,3),[],1);
end
size(X)

% apply Robust PCA
[n1,n2,n3] = size(X);
lambda = 1/sqrt(max(n1,n2)*n3*3);
opts.mu = 1e-5;
opts.tol = 1e-6;
opts.rho = 1.1;
opts.max_iter = 500;
opts.DEBUG = 1;
tic
[Xhat,E,err,iter] = trpca_tnn(X,lambda,opts);
toc

% prepare the new movie file
vidObj = VideoWriter('test_result.avi');
vidObj.FrameRate = frate;
open(vidObj);
range = 255;
map = repmat((0:range)'./range, 1, 3);
E(:,:,1) = medfilt2(E(:,:,1), [5,1]);
E(:,:,2) = medfilt2(E(:,:,2), [5,1]);
E(:,:,3) = medfilt2(E(:,:,3), [5,1]);% median filter in time
for i = (1:size(X, 1))
    frame1 = reshape(X(i,:,:),height,[], 3);
    frame2 = reshape(Xhat(i,:,:),height,[],3);
    frame3 = reshape(abs(E(i,:,:)),height,[],3);
    % median filter in space; threshold
    frame3(:,:,1) = (medfilt2(abs(frame3(:,:,1)), [5,5]) > 5).*frame1(:,:,1);
    frame3(:,:,2) = (medfilt2(abs(frame3(:,:,2)), [5,5]) > 5).*frame1(:,:,2);
    frame3(:,:,3) = (medfilt2(abs(frame3(:,:,3)), [5,5]) > 5).*frame1(:,:,3);
    % stack X, L and S together
    frame = cat(2, frame1, frame2, frame3);
    frame = uint8(frame);
    %frame = gray2ind(frame,range);
    %frame = im2frame(frame,map);
    writeVideo(vidObj,frame);
end
close(vidObj);