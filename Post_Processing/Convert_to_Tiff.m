fish = "fish_E";
parentFolder= "D:\Data\03302023_opto\Stacks";
%if ~exist(parentFolder, 'dir')
%   mkdir(parentFolder,fish);
%end

%set outpath
outpath = fullfile (parentFolder, fish);

% angle that you are going to deskew
delta = 119;

% conversion matrix
affineMatrix = [1 0 0 0; 0 1 0 0; 0 cotd(delta) 1 0 ; 0 0 0 1];

% apply affine
tform = affine3d(affineMatrix);
sizeMat = size(SCAPE_data); 

%create new matrix of the size of data
d_M = zeros(size(SCAPE_data(:,:,:,1)));

%rearrangement of the data for transformation

d_M = flip(flip(d_M,2),3);
size(d_M)
d_M = permute(d_M, [3,1,2]);
size(d_M)

% perform transformation
d_R = imref3d(size(d_M), 1, 3, 2);
[d_M, ~] = imwarp(d_M, d_R, tform);
d_M = permute(d_M, [1,2,3]);

% initialize voule to save transformed images
x = size(d_M);
SCAPE_volume = zeros(x(1),x(2),x(3),sizeMat(4));
size(SCAPE_volume)

% iterating over timepoint on timelapse
for c = 1:sizeMat(4)
   
    % Coordinate System Correction aka affine transform for each time point
    SCAPE_datax = flip(flip(SCAPE_data(:,:,:, c),2),3);
    SCAPE_datax = permute(SCAPE_datax, [3 1 2]);
    R = imref3d(size(SCAPE_datax), 1, 3, 2);
    [SCAPE_datax, ~] = imwarp(SCAPE_datax, R, tform);
    SCAPE_datax = permute(SCAPE_datax, [1 2 3]);
    
    %saving timepoint it into empty volume
    SCAPE_volume(:,:,:,c) = SCAPE_datax;
end

% saving data as zplanes through time for input into caiman
for d = 1:sizeMat(2)
    instance = squeeze(SCAPE_volume(:,:,d,:));    
    n = strcat(num2str(d), "_MatlabSkew.tiff");
    name = fullfile(outpath, n);
    name = convertStringsToChars(name);
    for j = 1:size(instance, 3)
        imwrite(uint16(instance(:,:,j)), name, 'tif', 'WriteMode', 'Append');
    end
    disp(d)
end



