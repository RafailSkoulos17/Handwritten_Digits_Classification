function diss = image2diss(image_set)
    
% Function used to create the dissimilarity representation from the
% image set. Initially, an image preprocessing step is taken and then
% the proxm function is applied on the dataset.
    
    preproc = im_box([],0)*im_box([],0,1)*im_resize([],[30 30])*im_box([],1,0);
    image_set = image_set*preproc;
    img_set = prdataset(image_set);
    mapping = proxm(img_set, 'd', 2);
    diss = img_set*mapping;
end
