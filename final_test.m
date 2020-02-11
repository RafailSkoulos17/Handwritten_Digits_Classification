% Script used to train the final classifiers and perform the nist_eval
% testing. Both scenarios are evaluated. In the first one classifiers
% trained on 200 samples per class are used on all representations (the best 
% classifer on each one of them) and in the second one classifiers
% trained on 10 samples per class are used on all representations (again the 
% best classifer on each one of them)

clear;
prwaitbar off;

disp('Loading the big datasets')
load('final_representations_200.mat'); % we didn't create the representations each time we just loaded them from
                                       % the matrix in which they were stored

disp('Classifier training for pixels')
clf = train_clf(pixel_rp, 'qdc', 'pcam');
disp('Evaluating')
e(1) = nist_eval('image2pixel', clf, 100);

disp('Classifier training for features')
clf = train_clf(features_rp, 'ldc', 'pcam');
disp('Evaluating')
e(2) = nist_eval('image2features', clf, 100);

disp('Classifier training for dissimilarity')
[trn_diss, val] = genddat(diss_rp, 0.5); 
clf = train_clf(trn_diss, 'qdc', 'psem');
disp('Evaluating')
e(3) = nist_eval('image2diss', clf, 100);

disp('Loading the small datasets')
load('final_representations_10.mat'); % we didn't create the representations each time we just loaded them from
                                      % the matrix in which they were stored

disp('Classifier training for pixels')
clf = train_clf(pixel_rp10, 'knn', 'pcam');
disp('Evaluating')
e(4) = nist_eval('image2pixel', clf);

disp('Classifier training for features')
clf = train_clf(features_rp10, 'vpc', 'pcam');
disp('Evaluating')
e(5) = nist_eval('image2features', clf);

disp('Classifier training for dissimilarity')
clf = train_clf(diss_rp10, 'qdc', 'psem');
disp('Evaluating')
e(6) = nist_eval('image2diss', clf);

save('final_results.mat','e');