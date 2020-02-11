% The script implementing the testing done for the classifiers created for
% the first scenario

clear;
prwarning off;

%% load the representations
load('final_representations_200.mat');

%% choose the representation 
a = 3;
if a == 1
    data = pixel_rp;
elseif a == 2
    data = features_rp;
else
    data = diss_rp;
end

%% split the dataset to training and test set
disp('Spliting the dataset to training validation and test sets')
if a ~= 3
    [trn, temp] = gendat(data, 0.4);
    [val, tst] = gendat(temp, 0.25);
else
    [trn, tst] = genddat(data, 0.4);
end

%% find the best dimension after pca to the training set
disp('Finding the dimensionality of pca or psem')
if a==1 || a==2
    best_pca = best_pcam_dim(trn);
    best_pca_with_val = best_pcam_dim(trn, val);
else
    best_psem = best_psem_dim(trn);
end

%% calculate the pca or psem mapping
disp('Calculating the pca or psem mapping')
if a ~= 3
    if best_pca ~= best_pca_with_val
        if best_pca_with_val == size(trn,2)
            best_pca_with_val = 50;
        end
        [pca_features, ] = pcam(trn, best_pca_with_val); 
    else
        if best_pca == size(trn,2)
            best_pca = 50;
        end
        pca_features = pcam(trn, best_pca);
    end    
else
    if best_psem == size(trn,2)
        best_psem = 50;
    end
    w_psem = psem(trn, best_psem);
end

%% Feature selection
disp('fevalf section to check the number of features to be used')
if a==1 || a==2
    maxJ = -1;
    start = size(pca_features,2)/10;
    step = start;
    endd = size(pca_features,2);
    for i=start:step:endd
        d = feateval(trn*pca_features(:,1:i), 'eucl-m');
        if maxJ < d
            maxJ = d;
            best_dim = i;
        end
    end
else
    maxJ = -1;
    start = size(w_psem,2)/10;
    step = start;
    endd = size(w_psem,2);
    for i=start:step:endd
        d = feateval(trn*w_psem(:,1:i), 'eucl-m');
        if maxJ < d
            maxJ = d;
            best_dim = i;
        end
    end
end
disp('Applying feature selection to the dataset')
if a==1 || a==2
    [featself_features, ] = featself(trn*pca_features, 'eucl-m', best_dim);
else
    [featself_features, ] = featself(trn*w_psem, 'eucl-m', best_dim);
end

%% Learning Curves section
disp('Learning curves plotting')
if a==1 || a==2
    disp('clevalf section')
    err_features = check_num_features(trn, tst, pca_features, featself_features);
    disp('cleval section')
    err_samples = check_num_samples(trn, tst, pca_features, featself_features);
    disp('cross validation section')
    [err_cross_val, hoerr] = evaluate_cl(tst*pca_features*featself_features, trn*pca_features*featself_features,...
        tst*pca_features*featself_features);
    disp('combiner section')
    [par_crossval_err, par_hoerr, seq_crossval_err, seq_hoerr] = combine_classifiers(tst*pca_features*featself_features,...
        trn*pca_features*featself_features, tst*pca_features*featself_features, 'parzenc', 'vpc', 'qdc');
else
    disp('clevalf section')
    err_features = check_num_features(trn, tst, w_psem, featself_features);
    disp('cleval section')
    err_samples = check_num_samples(trn, tst, w_psem, featself_features);
    disp('cross validation section')
    [err_cross_val, hoerr] = evaluate_cl(tst*w_psem*featself_features, trn*w_psem*featself_features,...
        tst*w_psem*featself_features);
    disp('combiner section')
    [par_crossval_err, par_hoerr, seq_crossval_err, seq_hoerr] = combine_classifiers(tst*w_psem*featself_features,...
        trn*w_psem*featself_features, tst*w_psem*featself_features, 'parzenc', 'knnc', 'qdc');
end