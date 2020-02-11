function pix_rp=image2pixel(image_set)

% Function used to create the pixel representation from the
% image set. Mostly image processing techniques have been applied to each
% image of the image_set separately.

sizeOfClass=size(image_set,1)/10;
labels = {{}};
for i=0:9
    for j=1:sizeOfClass
        image_ind = i*sizeOfClass + j;
        img = image_set(image_ind);
        bin_img = data2im(img);
        strl = strel('disk',1);
        preproc = imclose([], strl)*imerode([], strl)*imdilate([], strl)*im_box([],0,1);
        image = bin_img*preproc;
        centralMoments = im_moments(image,'central');
        orientation = atan(2*centralMoments(3)/(centralMoments(1)-centralMoments(2)));
        skew = affine2d([1 0 0; sin(0.5*pi-orientation) cos(0.5*pi-orientation) 0; 0 0 1]);
        image = imwarp(image, skew);
        image=im_resize(image,[30 30]);
        labels{image_ind} = strcat('digit_',num2str(i));
        result(image_ind,:)=image(:);
       
    end
end
 pix_rp = prdataset(result,labels');
end