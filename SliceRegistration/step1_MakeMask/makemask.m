imgpath1='F:\ztzhang\2d_3d_2022\fresh_DATai140_20220517\resgistration\CH1_scale-1.Labels.tif';
imgpath2='F:\ztzhang\2d_3d_2022\DAT_Ai140_LXN_20220318\registration\fixed_mask22.tif';
imgpath3='F:\ztzhang\2d_3d_2022\DAT_Ai140_LXN_20220318\registration\ann_new.tif';
for i = 1:229
    I1=uint8(imread(imgpath1,i));
%     I2=uint8(imread(imgpath2,i));
    I2=uint8(zeros(size(I1)));

    I2(I1==1)=10; %outline
    I2(I1==2)=60; %cp
    I2(I1==3)=20; %aco
    I2(I1==4)=140; % hip
    I2(I1==5)=110; %zi
    I2(I1==6)=160; %Pmv
    I2(I1==7)=250; %snc
    I2(I1==8)=210; %VTA
    I3=I2;
    I3(I2==10)=0;
    
%     I3=I2;
%     I2(I1==1)=10;
%     I2(I1==2)=140;
%     I3(I1==2)=140;
%     
%     I2(I1==3)=210;
%     I3(I1==3)=210; %V1
%     
%             %ZI
%     I2(I1==3)=110;
% %     I2(I1==967)=110;
%     %PMv
%     I2(I1==4)=160;
% %     I2(I1==1077)=160;
%     %VTA
%     I2(I1==5)=210;
% %     I2(I1==942)=210;
%     %SNc
%     I2(I1==6)=250;
%     I3=I2;
%     I3(I2==10)=0;
% %     I2(I1==774)=250;
%     
% %     
%         %ZI
%     I2(I1==379)=110;
%     I2(I1==967)=110;
%     %PMv
%     I2(I1==489)=160;
%     I2(I1==1077)=160;
%     %VTA
%     I2(I1==354)=210;
%     I2(I1==942)=210;
%     %SNc
%     I2(I1==186)=250;
%     I2(I1==774)=250;
    imwrite(I2,'./moving_mask22.tif','WriteMode','Append');
    imwrite(I3,'./moving_mask11.tif','WriteMode','Append');
end

for i = 1:1320
%     i
    I1=uint8(imread(imgpath2,i));
%     I3=uint8(imread(imgpath2,i));
    I2=uint16(imread(imgpath3,i));
%     I3=uint8(zeros(size(I1)));
    I3=uint8(zeros(size(I1)));
    I4=I3;
    I3(I1>0)=10;
    I3(I1==140)=140;
    I3(I1==20)=20;
    I3(I1==60)=60;
 
%     %V1;
%     gray=[17,152,275,347,369,391,605,740,863,935,957,979];
%     for j=1:12
%         I3(I2==gray(j))=210;
%         I4(I2==gray(j))=210;
%     end

       %ZI
    I3(I2==379)=110;
    I3(I2==967)=110;
    %PMv
    I3(I2==489)=160;
    I3(I2==1077)=160;
    %VTA
    I3(I2==354)=210;
    I3(I2==942)=210;
    %SNc
    I3(I2==186)=250;
    I3(I2==774)=250;
    I4=I3;
    I4(I3==10)=0;
% %     I2(I1==774)=250;
    
%     
%         %ZI
%     I2(I1==379)=110;
%     I2(I1==967)=110;
%     %PMv
%     I2(I1==489)=160;
%     I2(I1==1077)=160;
%     %VTA
%     I2(I1==354)=210;
%     I2(I1==942)=210;
%     %SNc
%     I2(I1==186)=250;
%     I2(I1==774)=250;
    imwrite(I3,'./registration1/fixed_mask2.tif','WriteMode','Append');
    imwrite(I4,'./registration1/fixed_mask1.tif','WriteMode','Append');
end
    