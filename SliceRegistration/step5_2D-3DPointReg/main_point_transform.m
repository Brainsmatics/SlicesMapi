clc;clear all
point_set=xlsread('./point_transform/220318.xlsx');
registration_path='F:/ztzhang/2d_3d_2022/DAT_Ai140_LXN_20220318/registration/registration1/';
linear1_path=[registration_path '/linear_mask2_250GenericAffine.mat'];
Warp1_path=[registration_path '/registration_mask1_250Warp.nii.gz'];
Warp2_path=[registration_path '/registration_mask2_250Warp.nii.gz'];
linear2_path=[registration_path '/registration_mask3_250GenericAffine.mat'];
Warp3_path=[registration_path '/registration_mask3_251Warp.nii.gz'];
point_set_new=point_set;

registration_x = 456;
registration_y = 360;
registration_z = 528;

scale=2.5;

disp('Load linear1 file ...');
linear1 = load(linear1_path);%线性参数1
disp('Load Warp1 file ...');
Warp1 = my_load_nii(Warp1_path);%非线性1
disp('Load Warp2 file ...');
Warp2 = my_load_nii(Warp2_path);%非线性2
disp('Load linear2 file ...');
linear2 = load(linear2_path);%线性参数2
disp('Load Warp3 file ...');
Warp3 = my_load_nii(Warp3_path);%非线性3
% disp('Load Warp4 file ...');
% Warp4 = my_load_nii([Src_registration '\adjust_251InverseWarp.nii.gz']);%非线性4

for i =1:length(point_set)
    x=point_set(i,2);
    y=point_set(i,3);
    z=point_set(i,4)*2;

    point=[x,y,z];
    inv=0;

    [New_Point_linear1] = linear_transform(point,linear1,1);
    [Nonlinear_Point1] = nonlinear_transform(New_Point_linear1,registration_x,registration_y,registration_z,Warp1,scale);
    [Nonlinear_Point2] = nonlinear_transform(Nonlinear_Point1,registration_x,registration_y,registration_z,Warp2,scale);
% %         % 第二次线性
    [New_Point_linear2] = linear_transform(Nonlinear_Point2,linear2,1);
    
    [Nonlinear_Point3] = nonlinear_transform(New_Point_linear2,registration_x,registration_y,registration_z,Warp3,scale);
    
    point_set_new(i,2)=Nonlinear_Point3(1);
    point_set_new(i,3)=Nonlinear_Point3(2);
    point_set_new(i,4)=Nonlinear_Point3(3);
end
% for i=1:length(point_set_new) 
%     if point_set_new(i,2)==1
%         point_set_new(i,5)=point_set_new(i,5)-80;
%     end
% end
point_set_save=point_set_new;

point_set_new=point_set_save;
Ann='F:\ztzhang\2d_3dregistration_master\single slice\volume_data\ann_new.tif';
for j =1:length(point_set_new)
    point_set_new(j,2)=ceil(point_set_new(j,2));
    point_set_new(j,3)=ceil(point_set_new(j,3));
    point_set_new(j,4)=ceil(point_set_new(j,4));
    [point_set_new(j,2),point_set_new(j,3),point_set_new(j,4),point_set_new(j,5)]=pointstrans(point_set_new(j,2),point_set_new(j,3),point_set_new(j,4),Ann);
% 	I(point_set_new(j,2),point_set_new(j,3))=255;
%             point_set_new(j,6)
%             point_set_new(j,3)
%             point_set_new(j,4)
%             point_set_new(j,5)
end
        

xlswrite('220318_point_trans_new.xlsx',point_set_new);
point_set_new=xlsread('220318_point_trans_new.xlsx');
point_set_save=point_set_new;
point_set_new=point_set_save;
for i=1:1320
    I=uint8(zeros(900,1140));
    Ann=uint16(imread('F:\ztzhang\2d_3dregistration_master\single slice\volume_data\ann_new.tif',i));
    for j = 1:length(point_set_new) 
        point_set_new(j,2)=ceil(point_set_new(j,2));
        point_set_new(j,3)=ceil(point_set_new(j,3));
        point_set_new(j,4)=ceil(point_set_new(j,4));
        if i==point_set_new(j,4)
            I(point_set_new(j,3),point_set_new(j,2))=255;
%             [point_set_new(j,4),point_set_new(j,3),point_set_new(j,6)]=pointstrans(point_set_new(j,4),point_set_new(j,3),Ann);
%             point_set_new(j,5)=Ann(point_set_new(j,2),point_set_new(j,3))
%             point_set_new(j,6)
%             point_set_new(j,3)
%             point_set_new(j,4)
%             point_set_new(j,5)
        end
        
    end
    imwrite(I,'point.tif','WriteMode','Append');
end