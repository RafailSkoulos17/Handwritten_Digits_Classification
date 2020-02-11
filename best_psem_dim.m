function best_psem_dim = best_psem_dim(trn)

% Function used to find the best number of dimensions for the PSEM mapping
% on the given training set trn. Only for dissimilarity matrices.

    maxJ = -1;
    for i=50:50:size(+trn,2)
        [w, ] = psem(trn, i);
        d = feateval(trn*w, 'eucl-m');
        if maxJ < d
            maxJ = d;
            best_psem_dim = i;
        end
    end
end