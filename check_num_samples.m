function fcs = check_num_samples(trn, tst, w_pca, w_sel)

% Function used to calculate the optimal number of training samples to be used.
% The classifiers evaluated are SVM, Nearest Mean, QDC, LDC, Parzen, KNN, 
% Logistic Linear, Fisher, Random Neural Net, Linear Perceptron and 
% Voted Perceptron. If w_pca and/or w_sel are specified then these two 
% mappings are applied on the datasets trn and tst to be used.

    if size(trn,1)>100 % check to change the start and step size for datasets of different size
        start = 100;
    else
        start = 10;
    end
    if nargin == 4
        fc_svc = cleval(trn*w_pca*w_sel,svc(proxm('e')),start:start:size(trn,1),1,...
        tst*w_pca*w_sel);
        fc_nmc = cleval(trn*w_pca*w_sel, nmc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_qdc = cleval(trn*w_pca*w_sel, qdc([],0, nan, []), start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_ldc = cleval(trn*w_pca*w_sel, ldc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_parzenc = cleval(trn*w_pca*w_sel, parzenc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_knnc = cleval(trn*w_pca*w_sel, knnc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_loglc = cleval(trn*w_pca*w_sel, loglc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_fisherc = cleval(trn*w_pca*w_sel, fisherc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_rnnc = cleval(trn*w_pca*w_sel, rnnc([],nan, nan), start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_perlc = cleval(trn*w_pca*w_sel, perlc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
        fc_vpc = cleval(trn*w_pca*w_sel, vpc, start:start:size(trn,1), 1,...
        tst*w_pca*w_sel);
    elseif nargin == 3
        fc_svc = cleval(trn*w_pca,svc(proxm('e')),start:start:size(trn,1),1, tst*w_pca);
        fc_nmc = cleval(trn*w_pca, nmc, start:start:size(trn,1), 1, tst*w_pca);
        fc_qdc = cleval(trn*w_pca, qdc([],0, nan, []), start:start:size(trn,1), 1, tst*w_pca);
        fc_ldc = cleval(trn*w_pca, ldc, start:start:size(trn,1), 1, tst*w_pca);
        fc_parzenc = cleval(trn*w_pca, parzenc, start:start:size(trn,1), 1, tst*w_pca);
        fc_knnc = cleval(trn*w_pca, knnc, start:start:size(trn,1), 1, tst*w_pca);
        fc_loglc = cleval(trn*w_pca, loglc, start:start:size(trn,1), 1, tst*w_pca); 
        fc_fisherc = cleval(trn*w_pca, fisherc, start:start:size(trn,1), 1, tst*w_pca);
        fc_rnnc = cleval(trn*w_pca, rnnc([],nan, nan), start:start:size(trn,1), 1, tst*w_pca);
        fc_perlc = cleval(trn*w_pca, perlc, start:start:size(trn,1), 1, tst*w_pca);
        fc_vpc = cleval(trn*w_pca, vpc, start:start:size(trn,1), 1, tst*w_pca);
    else
        fc_svc = cleval(trn,svc(proxm('e')),start:start:size(trn,1),1, tst);
        fc_nmc = cleval(trn, nmc, start:start:size(trn,1), 1, tst);
        fc_qdc = cleval(trn, qdc([],0, nan, []), start:start:size(trn,1), 1, tst);
        fc_ldc = cleval(trn, ldc, start:start:size(trn,1), 1, tst);
        fc_parzenc = cleval(trn, parzenc, start:start:size(trn,1), 1, tst);
        fc_knnc = cleval(trn, knnc, start:start:size(trn,1), 1, tst);
        fc_loglc = cleval(trn, loglc, start:start:size(trn,1), 1, tst);
        fc_fisherc = cleval(trn, fisherc, start:start:size(trn,1), 1, tst);
        fc_rnnc = cleval(trn, rnnc([],nan, nan), start:start:size(trn,1), 1, tst);
        fc_perlc = cleval(trn, perlc, start:start:size(trn,1), 1, tst);
        fc_vpc = cleval(trn, vpc, start:start:size(trn,1), 1, tst);
    end
    
    fcs = [fc_svc fc_nmc fc_qdc fc_ldc fc_parzenc fc_knnc fc_loglc fc_fisherc fc_rnnc fc_perlc fc_vpc];
    headers = ["svc","nmc","qdc" ,"ldc", "parzen", "knn", "log",...
               "fisher", "rnn", "perlc", "vpc"];    
    for i=1:11
        figure(i)        
        saveas(plote(fcs(i)),['plots/sample_size/'+headers(i)+'_result_per_sample_size_scen2.png']);
    end
end 