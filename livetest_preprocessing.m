function [other,labs,image] = livetest_preprocessing(image) 

% The function used to implement the main preprocessing steps to the image
% of the live test. It mainly tries to properly segment each digit in the
% image by identifying the sections of the image that are blank. More
% details on the procedure are presented in the report.

a = imread(image);  %Read the handwritten digits
b=rgb2gray(a);   %Convert RGB image to greyscale
BW = imbinarize(b); %Convert image to binary image
BW = imcomplement(BW);  
[width,length]=size(BW);
rows = [0 0 0 0 0 0 0 0 0]; % Matrix to save the lines that separate the handwritten digits by row
columns = [0 0 0 0 0 0 0 0 0]; % Matrix to save the lines that separate the handwritten digits by column
counter = 1;
line = [1 1];
col = [1 1];

for i=2:width
    for j=1:length
        if BW(i,j) == 1
            line(i) = 1;
        end       
    end   
    if length(line) < i
        line(i) = 0;
    end        
    if line(i-1) == 1 && line(i) == 0  % Check in which line there is only background 
        rows(counter) = i;
        counter = counter + 1;  
    end 
end

counter = 1;
for j=2:length
    for i=1:width
        if BW(i,j) == 1
            col(j) = 1;
        end
    end
    if length(col) < j
        col(j) = 0;
    end 
    if col(j-1) == 1 && col(j) == 0  % Check in which column there is only background 
        columns(counter) = j;
        counter = counter + 1;  
    end 
end

pr_mat = cell(100,1); % the cell array to hold the segmented digits as separate images
labs = cell(100,1); % the cell array to hold the label of each digit
cnt=1;
for i=1:length(rows)-1
    for j=1:length(columns)-1
        im = im_resize(BW(rows(i):rows(i+1), columns(j):columns(j+1)),[30,30]);
        pre = im_box([],0)*im_box([],0,1)*im_resize([],[30 30])*im_box([],1,0)*im_resize([],[30 30]);
        im = im * pre;
        pr_mat{cnt} = im;
        labs{cnt} = strcat('digit_',num2str(ceil(i-1)));
        cnt = cnt + 1;
    end
end 
other = prdataset(pr_mat, labs);
other.featsize = [30,30]; % setting the size of each picture in the dataset
image = cell(10);
for i = 1:10
        b = seldat(other,i);
    for j = 1:10
        c = data2im(b,j);
        image{i,j} = reshape(c,[30,30]);
    end
end
image = cell2mat(image); % all the segmented digits in a single image - mainly for self-evaluation reasons
end