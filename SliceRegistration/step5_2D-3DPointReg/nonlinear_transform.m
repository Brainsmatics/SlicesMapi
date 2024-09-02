% clc;clear;
% % 存在一个基本的问题，点会在形变场之外，解决办法只有扩大fixed，前后
% %% 基本参数
% Point = load('Point.swc');
% scale = 12.5;%缩放大小
% registration_x = 456;%形变场尺寸
% registration_y = 360;
% registration_z = 528;
%
%
% %% read warp_image
% % for i = 1:zsize
% %     Warpx(:,:,i) = imread('C1-registration_mask1_250Warp.nii.tif',i);
% %     Warpy(:,:,i) = imread('C2-registration_mask1_250Warp.nii.tif',i);
% %     Warpz(:,:,i) = imread('C3-registration_mask1_250Warp.nii.tif',i);
% % end



function [Nonlinear_Point] = nonlinear_transform(Point,registration_x,registration_y,registration_z,Warp,scale)
outpoint = 0;
Warpx = Warp(:,:,:,1,1);
Warpy = Warp(:,:,:,1,2);
Warpz = Warp(:,:,:,1,3);
%% 计算点重新赋值
%Point_num = length(Point);
[Point_num,Num_size] = size(Point);%点的数目
Nonlinear_Point = Point;
for i = 1:Point_num
    x = Point(i,1);
    y = Point(i,2);
    z = Point(i,3);
    new_y = y/scale;
    new_x = x/scale;
    new_z = z/scale;
    
    if (new_y<1 || new_y>registration_y || new_x<1 || new_x>registration_x || new_z<1 || new_z>registration_z)
        rx = x;
        ry = y;
        rz = z;
        outpoint = outpoint + 1;
        
    else
        %三线性插值
        new_z = 1-(y/scale-floor(y/scale));
        new_x = x/scale-floor(x/scale);
        new_y = z/scale-floor(z/scale);
        
        V2_x = Warpx(floor(y/scale),floor(x/scale),floor(z/scale));
        V6_x = Warpx(floor(y/scale),ceil(x/scale),floor(z/scale));
        V3_x = Warpx(floor(y/scale),floor(x/scale),ceil(z/scale));
        V7_x = Warpx(floor(y/scale),ceil(x/scale),ceil(z/scale));
        V0_x = Warpx(ceil(y/scale),floor(x/scale),floor(z/scale));
        V4_x = Warpx(ceil(y/scale),ceil(x/scale),floor(z/scale));
        V1_x = Warpx(ceil(y/scale),floor(x/scale),ceil(z/scale));
        V5_x = Warpx(ceil(y/scale),ceil(x/scale),ceil(z/scale));
        
        Vx = V0_x*(1-new_x)*(1-new_y)*(1-new_z) + V1_x*(1-new_x)*new_y*(1-new_z)+V2_x*(1-new_x)*(1-new_y)*new_z+V3_x*(1-new_x)*new_y*new_z+V4_x*new_x*(1-new_y)*(1-new_z)+V5_x*new_x*new_y*(1-new_z)+V6_x*new_x*(1-new_y)*new_z+V7_x*new_x*new_y*new_z;
        
        
        
        V2_y = Warpy(floor(y/scale),floor(x/scale),floor(z/scale));
        V6_y = Warpy(floor(y/scale),ceil(x/scale),floor(z/scale));
        V3_y = Warpy(floor(y/scale),floor(x/scale),ceil(z/scale));
        V7_y = Warpy(floor(y/scale),ceil(x/scale),ceil(z/scale));
        V0_y = Warpy(ceil(y/scale),floor(x/scale),floor(z/scale));
        V4_y = Warpy(ceil(y/scale),ceil(x/scale),floor(z/scale));
        V1_y = Warpy(ceil(y/scale),floor(x/scale),ceil(z/scale));
        V5_y = Warpy(ceil(y/scale),ceil(x/scale),ceil(z/scale));
        
        Vy = V0_y*(1-new_x)*(1-new_y)*(1-new_z) + V1_y*(1-new_x)*new_y*(1-new_z)+V2_y*(1-new_x)*(1-new_y)*new_z+V3_y*(1-new_x)*new_y*new_z+V4_y*new_x*(1-new_y)*(1-new_z)+V5_y*new_x*new_y*(1-new_z)+V6_y*new_x*(1-new_y)*new_z+V7_y*new_x*new_y*new_z;
        
        V2_z = Warpz(floor(y/scale),floor(x/scale),floor(z/scale));
        V6_z = Warpz(floor(y/scale),ceil(x/scale),floor(z/scale));
        V3_z = Warpz(floor(y/scale),floor(x/scale),ceil(z/scale));
        V7_z = Warpz(floor(y/scale),ceil(x/scale),ceil(z/scale));
        V0_z = Warpz(ceil(y/scale),floor(x/scale),floor(z/scale));
        V4_z = Warpz(ceil(y/scale),ceil(x/scale),floor(z/scale));
        V1_z = Warpz(ceil(y/scale),floor(x/scale),ceil(z/scale));
        V5_z = Warpz(ceil(y/scale),ceil(x/scale),ceil(z/scale));
        
        Vz = V0_z*(1-new_x)*(1-new_y)*(1-new_z)+V1_z*(1-new_x)*new_y*(1-new_z)+V2_z*(1-new_x)*(1-new_y)*new_z+V3_z*(1-new_x)*new_y*new_z+V4_z*new_x*(1-new_y)*(1-new_z)+V5_z*new_x*new_y*(1-new_z)+V6_z*new_x*(1-new_y)*new_z+V7_z*new_x*new_y*new_z;
        
        
        rx = x-Vx;
        ry = y-Vy;
        rz = z-Vz;
    end
    Nonlinear_Point(i,1) = rx;
    Nonlinear_Point(i,2) = ry;
    Nonlinear_Point(i,3) = rz;
end