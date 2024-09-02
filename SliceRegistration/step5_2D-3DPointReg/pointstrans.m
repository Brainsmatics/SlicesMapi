function [x,y,z,g]=pointstrans(x,y,z,im)
    gray=[354,489,186,942,1077,774,379,967];
    zz=[0,-1,1,-2,2,-3,3,-4,4,-5,5,-6,6,-7,7,-8,8,-9,9,-10,10];
    
    for k =1:21
        I=uint16(imread(im,z+zz(k)));
        for i=1:21
            for j =1:21
                if ismember (I(y+zz(j),x+zz(i)),gray)
                    x=x+zz(i);
                    y=y+zz(j);
                    z=z+zz(k);
                    g=I(y,x);
                    return
                end
            end
        end
    end
    I=uint16(imread(im,z));
    g=I(y,x);
end