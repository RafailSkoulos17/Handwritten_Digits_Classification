function clf = train_clf(data, cls_str, rdc_str)

% Function to apply the cls_str classifier and the rdc_str mapping on the
% dataset data and returns the trained classifier clf

    rdc = str2func(rdc_str);
    [pmap, ] = rdc(data, 50);
    [fs, ] = featself(data*pmap, 'eucl-m', 50);
    cl = str2func(cls_str);
	cls = cl(data*pmap*fs);
    clf = pmap*fs*cls;
end