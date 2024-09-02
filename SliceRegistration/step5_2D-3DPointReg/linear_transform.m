% clear all;clear;
% %% ��������
% load('E:\�ձ�\��2017_05_11 �������������ݼ���Ԫʸ���ļ��ı任\fucntion\Point.swc');% ���ļ�
% load('E:\�ձ�\��2017_05_11 �������������ݼ���Ԫʸ���ļ��ı任\14193\linear_nonlinear_para\1\linear_mask2_250GenericAffine.mat');%������׼�ļ�
%
%
% scale = 12.5; %�Ŵ���

function [New_Point_linear] = linear_transform(Point,linear,scale)
%% ������׼����
transf=linear.AffineTransform_double_3_3;
% transf=AffineTransform_float_3_3;
center=linear.fixed;

M=[transf(1) transf(2) transf(3)
    transf(4) transf(5) transf(6)
    transf(7) transf(8) transf(9)];
translation=[transf(10) transf(11) transf(12)];
for i=1:3
    offset(i)=translation(i)+center(i);
    for j=1:3
        offset(i) = offset(i)-M(i,j) * center(j);
    end
end

M1=[transf(1) transf(2) transf(3)  offset(1)
    transf(4) transf(5) transf(6)  offset(2)
    transf(7) transf(8) transf(9)  offset(3)
    0 0 0 1];
M1=M1';
M1=inv(M1);

%% ���ž���-����С���طŴ�
% ��С����
scale1 = [1/scale 0 0 0
    0 1/scale 0 0
    0 0 1/scale 0
    0 0 0 1];

% �Ŵ����
scale2 = [scale 0 0 0
    0 scale 0 0
    0 0 scale 0
    0 0 0 1];

M1 = scale1*M1*scale2;% �¾���

%% �����µĵ㼯������
%Num_Point = length(Point);
[Num_Point,Num_size] = size(Point);%�����Ŀ
New_Point_linear = Point;
for i = 1:Num_Point
%     i;
    x = Point(i,1);
    y = Point(i,2);
    z = Point(i,3);
    P_warp = [x y z 1]*M1;
    %     P_warp = [x y z 1]*rot_martix;
    New_Point_linear(i,1) = P_warp(1);
    New_Point_linear(i,2) = P_warp(2);
    New_Point_linear(i,3) = P_warp(3);
end
