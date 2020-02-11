function [par_crossval_err, par_hoerr, seq_crossval_err, seq_hoerr] = ...
    combine_classifiers(data, trn, tst, cl1, cl2, cl3)

% Function used to construct and evalueate the combination of classifiers
% cl1, cl2 and cl3 both in a sequential and in a parallel structure. The
% whole dataset data is used for crossvalidation while the sets trn and tst
% are used for the calculation of the holdout error.

cl1 = str2func(cl1);
cl2 = str2func(cl2);
cl3 = str2func(cl3);

n_cl1 = cl1([]);
n_cl2 = cl2([]);
n_cl3 = cl3([]);

headers = ["Cmax","Cmin","Cmean" ,"Cprod"];

%% -------------Parallel combiner-------------

W = [n_cl1 n_cl2 n_cl3];

Cmax = W*maxc;            % max combiner
Cmin = W*minc;            % min combiner
Cmean = W*meanc;          % mean combiner
Cprod = W*prodc;          % product combiner

disp('-----------Crossvalidation----------')
par_crossval_err = prcrossval(data,{Cmax,Cmin,Cmean,Cprod}, 10);

par_crossval_err_results = [headers; par_crossval_err];
    
save('results/par_comb_crosseval_results_scen2.mat','par_crossval_err_results');

disp('-----------Hold-out----------')
cls1 = Cmax(trn); par_hoerr(1) = testc(tst*cls1);
cls2 = Cmin(trn); par_hoerr(2) = testc(tst*cls2);
cls3 = Cmean(trn); par_hoerr(3) = testc(tst*cls3);
cls4 = Cprod(trn); par_hoerr(4) = testc(tst*cls4);

par_hoerr_results = [headers; par_hoerr];
    
save('results/par_comb_holdouteval_results_scen2.mat','par_hoerr_results');


%% -------------Sequentional combiner-------------
W = [n_cl1*classc n_cl2*classc]*n_cl3;

Cmax = W*maxc;            % max combiner
Cmin = W*minc;            % min combiner
Cmean = W*meanc;          % mean combiner
Cprod = W*prodc;          % product combiner

disp('-----------Crossvalidation----------')
seq_crossval_err = prcrossval(data,{Cmax,Cmin,Cmean,Cprod}, 10);

seq_crossval_err_results = [headers; seq_crossval_err];
save('results/seq_comb_crosseval_results_scen2.mat','seq_crossval_err_results');

disp('-----------Hold-out----------')
cls1 = Cmax(trn); seq_hoerr(1) = testc(tst*cls1);
cls2 = Cmin(trn); seq_hoerr(2) = testc(tst*cls2);
cls3 = Cmean(trn); seq_hoerr(3) = testc(tst*cls3);
cls4 = Cprod(trn); seq_hoerr(4) = testc(tst*cls4);

seq_hoerr_results = [headers; seq_hoerr];
save('results/seq_comb_holdouteval_results_scen2.mat','seq_hoerr_results');

end

