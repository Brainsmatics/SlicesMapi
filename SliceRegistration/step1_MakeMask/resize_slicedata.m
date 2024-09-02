x0=0;
y0=0;
for i = 3:460
imgpath=['F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\Preview\220517_preview_' num2str(i,'%05d') '_CH1.jpg'];
I=imread(imgpath);
[y,x]=size(I);
if y>y0
    y0=y;
end
if x>x0
    x0=x;
end
end

savepath11='F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH1\';
savepath12='F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH1_scale\';
savepath21='F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH2\';
savepath22='F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH2_scale\';
mkdir(savepath11);
mkdir(savepath12);
mkdir(savepath21);
mkdir(savepath22);

% x0=8748;
% y0=6656;
parfor i = 3:460
imgpath1=['F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\Preview\220517_preview_' num2str(i,'%05d') '_CH1.jpg'];
savepath11=['F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH1\220517_preview_' num2str(i,'%05d') '_CH1.tif'];
savepath12=['F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH1_scale\220517_preview_' num2str(i,'%05d') '_CH1.tif'];

imgpath2=['F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\Preview\220517_preview_' num2str(i,'%05d') '_CH2.jpg'];
savepath21=['F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH2\220517_preview_' num2str(i,'%05d') '_CH2.tif'];
savepath22=['F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH2_scale\220517_preview_' num2str(i,'%05d') '_CH2.tif'];

I1=imread(imgpath1);
I2=imread(imgpath2);
[y,x]=size(I1);
I11=uint8(zeros([y0,x0]));
I21=uint8(zeros([y0,x0]));
xt=floor((x0-x)/2);
yt=floor((y0-y)/2);
I11(yt+1:yt+y,xt+1:xt+x)=I1;
I21(yt+1:yt+y,xt+1:xt+x)=I2;
I12=imresize(I11,0.2);
I22=imresize(I21,0.2);
imwrite(I11,savepath11);
imwrite(I12,savepath12);
imwrite(I21,savepath21);
imwrite(I22,savepath22);

end