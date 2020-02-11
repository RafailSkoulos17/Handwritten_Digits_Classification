% Script used to create all the representations from the NIST dataset for
% both scenarios and store them in matrices for further processing.

clear;
prwaitbar off; 

% take 200 images per class
disp('Obtaining the 200 images per class')
image_set = prnist(0:9, 1:5:1000);

% create the representations
disp('Creating the representations')
disp('Pixel Representation...')
pixel_rp = image2pixel(image_set);
disp('Features Representation')
features_rp = image2features(image_set);
disp('Dissimilarity Representation')
diss_rp = image2diss(image_set);
save('final_representations_200', 'pixel_rp', 'features_rp', 'diss_rp');

% take 10 images per class
disp('Obtaining the 10 images per class')
image_set10 = prnist(0:9, 1:100:1000);

% create the representations
disp('Creating the representations')
disp('Pixel Representation...')
pixel_rp10 = image2pixel(image_set10);
disp('Features Representation')
features_rp10 = image2features(image_set10);
disp('Dissimilarity Representation')
diss_rp10 = image2diss(image_set10);
save('final_representations_10', 'pixel_rp10', 'features_rp10', 'diss_rp10');