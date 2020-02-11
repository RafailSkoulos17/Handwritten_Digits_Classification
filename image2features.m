function feat_rp = image2features(image_set)

% Function used to create the feature representation from the
% image set. Initially, an image preprocessing step is taken and then
% multiple types of features are extracted from the dataset.

    preproc = im_box([],0)*im_box([],0,1)*im_resize([],[30 30])*im_box([],1,0);
    image_set = image_set*preproc;
    img_set = prdataset(image_set);
    feat_im = im_features(img_set, img_set, 'all');
    feat_moments = im_moments(img_set, 'central');
    feat_profile = im_profile(img_set);
    feat_mean = im_mean(img_set);
    feat_stat = im_stat(img_set);
    feat_skel = im_skel_meas(img_set);
    norm_feats = normalize([+feat_im +feat_moments +feat_profile +feat_mean +feat_stat +feat_skel],'range');
    feat_rp = prdataset(norm_feats, getlabels(feat_im));
end