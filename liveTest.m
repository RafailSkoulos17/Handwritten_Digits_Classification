% Script used to implement the final live test. The current version of the
% script is for the second scenario.

clear;
prwaitbar off;
disp('Taking the images set for live testing')
[images_set, labels, img] = livetest_preprocessing('digits.png');
disp('Images to dataset')
live_tst = image2pixel_live(images_set);
disp('Classifier training')
load('final_representations_10.mat');
clf = train_clf(pixel_rp10, 'nmc', 'pcam');
disp('Evaluating')
e = testc(live_tst*clf);