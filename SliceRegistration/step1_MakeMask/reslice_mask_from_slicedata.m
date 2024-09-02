imgpath1='F:\ztzhang\2d_3dregistration_master\single slice\volume_data\SliceData2\ann_8.97_-4.07.tif';
imgpath2='F:\ztzhang\2d_3dregistration_master\single slice\volume_data\SliceData2\fixed_8.97_-4.07.tif';
for i = 700:850
    I1=uint16(imread(imgpath1,i));
    I3=uint8(imread(imgpath2,i));
    I2=uint8(zeros(size(I3)));

    I2(I3>0)=10;
    I2(I3==140)=140;
    I2(I3==60)=60;
        %ZI
    I2(I1==379)=110;
    I2(I1==967)=110;
    %PMv
    I2(I1==489)=160;
    I2(I1==1077)=160;
    %VTA
    I2(I1==354)=210;
    I2(I1==942)=210;
    %SNc
    I2(I1==186)=250;
    I2(I1==774)=250;
    imwrite(I1,'ann_700-850.tif','WriteMode','Append');
    imwrite(I2,'fixed_700-850.tif','WriteMode','Append');
end
    