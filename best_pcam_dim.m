function best_pca_dim = best_pcam_dim(trn, val)

% Function used to find the best number of dimensions for the pca method 
% on the given training set trn. If also a second argument is provided
% then a validation set is also used in the feateval method

    maxJ = -1;
    for i=50:50:size(+trn,2)
        [pmap, ] = pcam(trn, i);
        if nargin == 1
            d = feateval(trn*pmap, 'eucl-m');
        else
            d = feateval(trn*pmap, 'eucl-m', val*pmap);
        end
        if maxJ < d
            maxJ = d;
            best_pca_dim = i;
        end
    end
end