function pix_rp=image2pixel_live(image_set)

% Function used to create the pixel representation for the live test 
% image set. Some extra image processing techniques have been applied to each
% image of the image_set comparing to the image2pixel function.

sizeOfClass=size(image_set,1)/10;
labels = {{}};
for i=0:9
    for j=1:sizeOfClass
        image_ind = i*sizeOfClass + j;
        img = reshape(+image_set(image_ind,:),[30,30]);
        strl = strel('disk',1);
        preproc = imclose([], strl)*imerode([], strl)*imdilate([], strl)*im_box([],0,1);
        image = img*preproc;
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