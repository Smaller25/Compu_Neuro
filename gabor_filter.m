%% Gabor Filtering
% 이미지 내의 영역을 텍스처를 기준으로 분할하는데 사용되는 가보르 필터의 응용
% Gabor filter :  선형필터, 일반적으로는 영상에서 특정 방향의 성분 찾음 / 사인함수로 모델링된 가우시안
    % Bio inspired 키워드에선 거의 각설이라고 한다
    % 포유류 시각 인지 시스템의 단순 세포(simple cell)의 작동 방식을 모델링한 필터
% A. K. Jain and F. Farrokhnia, "Unsupervised Texture Segmentation Using Gabor Filters",1991
% https://kr.mathworks.com/help/images/texture-segmentation-using-gabor-filters.html
%----------------------------
% Image Loading
A = imread('kobi.png');
A = imresize(A,0.25);
Agray = rgb2gray(A);

figure
imshow(A)

% Gabor filter 생성
image_size = size(A);
n_rows = size(A,1);
n_cols = size(A,2);

wave_length_min = 4/(sqrt(2));
wave_length_max = hypot(n_rows,n_cols);
n = floor(log2(wave_length_max/wave_length_min));
wave_length = 2.^(0:(n-2)) * wave_length_min;

d_theta = 45;
orientation = 0:d_theta : (180-d_theta);

g = gabor(wave_length, orientation);

% gabor filter 적용
gabor_mag = imgaborfilt(Agray, g);

% gabor property 이용한 후처리
% 1) gauss 저역 통과 필터 이용해서 경계 부드럽게 + 평활화
for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    K = 3;
    gabor_mag(:,:,i) = imgaussfilt(gabor_mag(:,:,i),K*sigma);
end

% 2) 공간 정보 (x,y) 추가 -> 가까운 영역끼리 묶이게끔 공간 정보 필요
x = 1:n_cols;
y = 1:n_rows;
[x, y] = meshgrid(x,y);
feature_set = cat(3, gabor_mag, x);
feature_set = cat(3, feature_set, y);

% 3) data 정리 : kmeans 함수 쓰기 전에 형식 맞추기 / feature_set engineering
n_points = n_rows*n_cols;
X = reshape(feature_set, n_rows*n_cols,[]);
% 평균 0, 분산 1 되도록 정규화
X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, std(X));

% 시각화
coeff = pca(X);
feature2DImage = reshape(X*coeff(:,1),n_rows,n_cols);
figure
imshow(feature2DImage,[])

%-----------------------------
% kmeans 이용해서 텍스처 분류하기
L = kmeans(X,2,'Replicates',5);
L = reshape(L,[n_rows, n_cols]); 
figure 
imshow(label2rgb(L))

% 시각화
Aseg1 = zeros(size(A),'like',A);
Aseg2 = zeros(size(A),'like',A);
BW = L == 2;
BW = repmat(BW,[1 1 3]);
Aseg1(BW) = A(BW);
Aseg2(~BW) = A(~BW);
figure
imshowpair(Aseg1,Aseg2,'montage');