% img = read('img/png')

function img = read(way)
img = {};
a = dir (way); % 'img/png'
a1 = struct('folder', {a(1:end).folder});
a2 = (struct('name', {a(1:end).name}));
for  i =3:length(a1)
x1 = a1(i);
x2 = a2(i);
patch = string(x1.folder)+"\"+string(x2.name);    
% img{i-2}=(imresize(double(imread(patch)),[50,50]));
img{i-2}=(double(imread(patch)));
end

% vet_files=dir('img/png/*.png');
% for i=3:length(a1)
%     I=imread(sprintf('RBSvideo/%s',vet_files(i).name));
% end

% T=rand*10*randi([-1,1],1,1);
% tform = maketform('affine', [cosd(T) -sind(T) 0; sind(T) cosd(T) 0; 0 0 1]); 
% imgData =dlarray(imtransform(img,tform,'size',size(img),'fill',255));
% imgData = C_B_V(fix(imgData+(rand*randi([-50,50],1,1))));
% imgData(imgData==(max(max(max(imgData)))))=255;
end
% function x=C_B_V(x)
% x(x>255)=255; 
% x(x<0)=0; 
% end