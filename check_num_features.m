function fcs = check_num_features(trn, tst, w_pca, w_sel)

% Function used to calculate the optimal number of features to be used.
% The classifiers evaluated are SVM, Nearest Mean, QDC, LDC, Parzen, KNN, 
% Logistic Linear, Fisher, Random Neural Net, Linear Perceptron and 
% Voted Perceptron. If w_pca and/or w_sel are specified then these two 
% mappings are applied on the datasets trn and tst to be used.

    if nargin == 4
        endd = min(50, size(w_sel,2));
        fc_svc = clevalf(trn*w_pca*w_sel,svc(proxm('e')),5:5:endd,[],1,...
        tst*w_pca*w_sel);
        fc_nmc = clevalf(trn*w_pca*w_sel, nmc, 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_qdc = clevalf(trn*w_pca*w_sel, qdc([],0, 0, nan), 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_ldc = clevalf(trn*w_pca*w_sel, ldc([],0, 0, nan), 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_parzenc = clevalf(trn*w_pca*w_sel, parzenc, 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_knnc = clevalf(trn*w_pca*w_sel, knnc, 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_loglc = clevalf(trn*w_pca*w_sel, loglc, 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_fisherc = clevalf(trn*w_pca*w_sel, fisherc, 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_rnnc = clevalf(trn*w_pca*w_sel, rnnc([],nan, nan), 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_perlc = clevalf(trn*w_pca*w_sel, perlc, 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
        fc_vpc = clevalf(trn*w_pca*w_sel, vpc, 5:5:endd, [], 1,...
        tst*w_pca*w_sel);
    elseif nargin == 3
        endd = min(50, size(w_pca,2));
        fc_svc = clevalf(trn*w_pca,svc(proxm('e')),5:5:endd,[],1, tst*w_pca);
        fc_nmc = clevalf(trn*w_pca, nmc, 5:5:endd, [], 1, tst*w_pca);
        fc_qdc = clevalf(trn*w_pca, qdc([],0, 0, nan), 5:5:endd, [], 1, tst*w_pca);
        fc_ldc = clevalf(trn*w_pca, ldc([],0, 0, nan), 5:5:endd, [], 1, tst*w_pca);
        fc_parzenc = clevalf(trn*w_pca, parzenc, 5:5:endd, [], 1, tst*w_pca);
        fc_knnc = clevalf(trn*w_pca, knnc, 5:5:endd, [], 1, tst*w_pca);
        fc_loglc = clevalf(trn*w_pca, loglc, 5:5:endd, [], 1, tst*w_pca);
        fc_fisherc = clevalf(trn*w_pca, fisherc, 5:5:endd, [], 1, tst*w_pca);
        fc_rnnc = clevalf(trn*w_pca, rnnc([],nan, nan), 5:5:endd, [], 1, tst*w_pca);
        fc_perlc = clevalf(trn*w_pca, perlc, 5:5:endd, [], 1, tst*w_pca);
        fc_vpc = clevalf(trn*w_pca, vpc, 5:5:endd, [], 1, tst*w_pca);
    else
        fc_svc = clevalf(trn,svc(proxm('e')),5:5:50,[],1, tst);
        fc_nmc = clevalf(trn, nmc, 5:5:50, [], 1, tst);
        fc_qdc = clevalf(trn, qdc([],0, 0, nan), 5:5:50, [], 1, tst);
        fc_ldc = clevalf(trn, ldc([],0, 0, nan), 5:5:50, [], 1, tst);
        fc_parzenc = clevalf(trn, parzenc, 5:5:50, [], 1, tst);
        fc_knnc = clevalf(trn, knnc, 5:5:50, [], 1, tst);
        fc_loglc = clevalf(trn, loglc, 5:5:50, [], 1, tst);
        fc_fisherc = clevalf(trn, fisherc, 5:5:50, [], 1, tst);
        fc_rnnc = clevalf(trn, rnnc([],nan, nan), 5:5:50, [], 1, tst);
        fc_perlc = clevalf(trn, perlc, 5:5:50, [], 1, tst);
        fc_vpc = clevalf(trn, vpc, 5:5:50, [], 1, tst);
    end
    
    fcs = [fc_svc fc_nmc fc_qdc fc_ldc fc_parzenc fc_knnc fc_loglc fc_fisherc fc_rnnc fc_perlc fc_vpc];
    
    headers = ["svc","nmc","qdc" ,"ldc", "parzen", "knn", "log",...
               "fisher", "rnn", "perlc", "vpc"];   
    for i=1:11
        figure(i)
        saveas(plote(fcs(i)),['plots/learning_curve/'+headers(i)+'_learning_curve_scen2.png']);
    end
end 