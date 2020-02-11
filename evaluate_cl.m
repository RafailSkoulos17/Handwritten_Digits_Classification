function [err, hoerr] = evaluate_cl(data, trn, tst)   

% Function used to perform crossvalidation and calulate the holdout error.
% The classifiers evaluated are SVM, Nearest Mean, QDC, LDC, Parzen, KNN, 
% Logistic Linear, Fisher, Random Neural Net, Linear Perceptron and 
% Voted Perceptron. The dataset data is used for the crossvalidation part
% is data and the sets trn and tst are used for the holdout error. 

    err(1)= prcrossval(data,svc([],proxm('e')),10);
    err(2)= prcrossval(data,nmc,10);
    err(3)= prcrossval(data,qdc([],0, nan, []),10);
    err(4)= prcrossval(data,klldc,10);
    err(5)= prcrossval(data,parzenc,10);
    err(6)= prcrossval(data,knnc,10);
    err(7)= prcrossval(data,logmlc,10);
    err(8)= prcrossval(data,fisherc,10);
    err(9)= prcrossval(data,rnnc([], nan, nan),10);
    err(10)= prcrossval(data,perlc,10);
    err(11)= prcrossval(data,vpc,10);

    headers = ["svc","nmc","qdc" ,"ldc", "parzen", "knn", "log",...
               "fisher", "rnn", "perlc", "vpc"];
      
    crossval_results = [headers; err];

    save('results/crossval_results_scen2.mat','crossval_results');
    
    cls1 = svc(trn,proxm('e')); hoerr(1) = testc(tst*cls1);
    cls2 = nmc(trn); hoerr(2) = testc(tst*cls2);
    cls3 = qdc(trn,0, nan, []); hoerr(3) = testc(tst*cls3);
    cls4 = klldc(trn); hoerr(4) = testc(tst*cls4);
    cls5 = parzenc(trn); hoerr(5) = testc(tst*cls5);
    cls6 = knnc(trn); hoerr(6) = testc(tst*cls6);
    cls7 = logmlc(trn); hoerr(7) = testc(tst*cls7);
    cls8 = fisherc(trn); hoerr(8) = testc(tst*cls8);
    cls9 = rnnc(trn, nan, nan); hoerr(9) = testc(tst*cls9);
    cls10 = perlc(trn); hoerr(10) = testc(tst*cls10);
    cls11 = vpc(trn); hoerr(11) = testc(tst*cls11);

    headers = ["svc","nmc","qdc" ,"ldc", "parzen", "knn", "log",...
               "fisher", "rnn", "perlc", "vpc"];
          
    results = [headers; hoerr];
    
    save('results/eval_results_scen2.mat','results');

end