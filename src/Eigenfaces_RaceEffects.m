
% function [] = runsss()  
%------------------------------------------------------
% read a CSV  file of face images, with one face per column.
% Matlab's readtable() will do this.
% The first row of the CSV file is the header row, which contains the names of the columns.
% The first column of the CSV file is the row index, which is not needed.

% AllFaces  = readtable('/Users/aakim/ProjectsMatlab/SingularValueDecomposition/EigenFaces/all_images_CFD1142.csv');          % headers become T.Properties.VariableNames
% headers = AllFaces.Properties.VariableNames;
% AllFacesMatrix = table2array(AllFaces);                 % numeric matrix, headers stored separately

%-----------------------------------------------------
% Calculate the average face for the training set of faces, and plot it. 
%-
% Matrix trainingFaces is the faces 2-59 (AF faces)
%%trainingFaces = AllFacesMatrix(:,2:59);        % AF faces
%trainingFaces = AllFacesMatrix(:,60:111);      % AM faces
%trainingFaces = AllFacesMatrix(:,112:215);     % BF faces  
%%trainingFaces = AllFacesMatrix(:, 216:308);   % BM faces
%trainingFaces = AllFacesMatrix(:, 309:401);     % CF faces
all_faces_file = '/Users/aakim/ProjectsMatlab/SingularValueDecomposition/EigenFaces/all_images_CFD1142.csv';

startcol = 2;  % first face column to read in (AF faces)
endcol = 59;  % last face column to read in (AF faces)

trainingFaces = read_face_data(all_faces_file, startcol, endcol)  
nFaces = size(trainingFaces, 2);  % number of faces in the training set

% Each face image is a column vector of values 1-255, which are the pixel values of the images.
% The images are 224 x 224, so each column vector has 224*224 = 50176 values.
% reshape column vector into 224 x 224 matrix
nrows = 224; 
ncols = 224;

%% --- Step 1: Mean-center the data ---
% PCA looks for directions of maximum variance, so we must first
% remove the average face; otherwise the dominant "direction" found
% by SVD would just be something like "faces have a bright blob in the middle",
% That is, it would mostly describe the mean itself rather than how faces
% differ from one another.
%
% We'll mean center by subtract the average face from every training face.
% Note: The column vector meanFace will be implicitly expanded to a matrix of the same size as trainingFaces, 
% with each column equal to meanFace.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the average face. 
meanFace = mean(trainingFaces,2);  % size (nrows * ncols) by 1;
A = trainingFaces - meanFace;  % subtract the average face from each training face.  A is the mean-centered training data matrix.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute eigenfaces on mean-subtracted training data matrix A.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Step 2: Singular Value Decomposition ---
% Any matrix A (m x n) can be factored as A = U * S * V', where:
%   U (m x m)  - orthonormal columns spanning the column space of A
%                (here: "pixel space" directions -> the eigenfaces)
%   S (m x n)  - diagonal, singular values sigma_1 >= sigma_2 >= ... >= 0,
%                describing how much "spread" the data has along each
%                corresponding direction in U
%   V (n x n)  - orthonormal columns spanning the row space of A
%                (here: how much each face "uses" each eigenface,
%                up to scaling by S)
%
%   **Note**:  The columns of U are the eigenfaces.
% Relationship to PCA: the columns of U are exactly the eigenvectors of
% the covariance matrix (1/(n-1)) * A * A', and the eigenvalues of that
% covariance matrix are singularValues.^2 / (n-1). Computing the SVD of
% A directly avoids ever forming A*A' (which would be an enormous
% 50176 x 50176 matrix) -- this is the main numerical/memory advantage
% of using svd() over eig(cov(faces')).
%
% run the 'econ' (economy-size) SVD: this will save memory and compute.
[U, S, V] = svd(A, 'econ');   % run the SVD. This is the key computation!
singularValues = diag(S);  % the singular values are the diagonal elements of S.
% Convert singular values to variance ("eigenvalues" of the covariance
% matrix) explained by each component. This follows from the fact that
% the covariance matrix C = A*A'/(n-1) has eigenvalues sigma_i^2/(n-1)
% when A = U*S*V' is the SVD of the centered data.
eigenvalues = singularValues.^2 / (nFaces - 1);   % variance along each PC
 % Normalize so we can talk about "percent of variance explained" by
% each component, and how that accumulates as we add more components.
varExplained = eigenvalues / sum(eigenvalues);
cumVarExplained = cumsum(varExplained);

% ------------------------------------------------------------------------------------
%% --- Project faces into eigenface space ---
% Each column of U is an eigenface, and the columns of U form an orthonormal basis for the space of faces.
% Because the columns of U are orthonormal, projecting a centered face
% onto the eigenface basis is just a matrix multiply by U'. 
% The rows of U' are the columns of U (the eigenfaces), 
% so the matrix multiplication will calculate the inner product of each training face (a column of A)
% with each eigenface (a column of U), 
% yielding a coordinate that describes how much of that eigenface is present in the training face.
% Each cell of weights is one of these inner products of a training face with an eigenface.
% 
% The weights matrix is the coordinates of each face in the eigenface basis.
% The columns of the weights matrix correspond to the faces, 
% and the rows correspond to the eigenfaces, which are ordered by the amount of variance they explain (largest singular value first).
% 
% weights(:,i) is the i-th face's coordinates in eigenface space
weights = U' * A;                       
% Note of interest:  Equivalently, we could do weights = S * V' 
% this outs to the same thing as U' * A, given
% the SVD identity A = U*S*V'), but computing it via U'*A is the most
% direct/intuitive: "how much does face i overlap with eigenface k".
% ------------------------------------------------------------------------------------
% Reconstruct Faces from Eigenface Weights
% Any mean-centered face can be reconstructed exactly (using ALL components) as: 
% A(:,i) = U * weights(:,i)
% And if we want to reconstruct the original, uncentered face, we add the mean face back in:
% reconstructed_faces(:,i) = meanFace + U*weights(:,i)

%
% Sanity check: reconstruct face 1 using the full basis and confirm we
% recover it exactly (up to floating-point round-off).
% ------------------------------------------------------------------------------------

reconTest = meanFace + U * weights(:,1);
fprintf('Max reconstruction error (face 1, full basis): %g\n', ...
    max(abs(reconTest - trainingFaces(:,1))));

%% --- Reconstruct a face using only the top dim_reduced components ---
% Instead of using all the eigenfaces, use only the first few eigenfaces (the most
% important ones). This approximates the original face using far less
% information: meanFace + (sum of k weighted eigenfaces), instead of
% meanFace + (sum of all weighted eigenfaces).
dim_reduced = 3;
faceIdx = 1;   % which face (column) to demonstrate reconstruction on
reconK = meanFace + U(:,1:dim_reduced) * weights(1:dim_reduced, faceIdx);
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize Results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot the average face.
figure, axes('position',[0  0  1  1]), axis off
imagesc(reshape(meanFace, nrows, ncols)), colormap gray; title('The Average Face');

% Plot the top eigenfaces: reshape each of the first nShow columns of U back
% into a 224x224 image. 
% These are NOT real faces -- they're the "directions of variation", 
% so they often look like abstract blends of
% light/dark patches (glasses, facial hair, lighting direction, etc.).
figure('Name', 'Top 32 Eigenfaces');
nShow = 32;
for i = 1:nShow
    subplot(4, 8, i);
    %%figure; imagesc(EigenFaces), colormap gray    
    imagesc(reshape(U(:,i), nrows, ncols)); axis image off; colormap gray;
    %%imagesc(reshape(U(:,i), nrows, ncols)); colormap gray;
    title(sprintf('PC %d (%.1f%%)', i, 100*varExplained(i)));
end


% project the a new face onto the first five eigenfaces and reconstruct it using only those five eigenfaces.
%%newFace = AllFacesMatrix(:, 60);  % pick a face from the training set (AF face)


% Scree plot / cumulative variance: shows how quickly adding more
% components captures the total variance in the dataset. A steep drop
% early on (typical for faces) means most of the "information" lives in
% just a few principal components, which is what makes eigenfaces such
% an effective compression technique.
figure('Name', 'Variance Explained');
subplot(1,2,1);
plot(varExplained, 'o-'); xlabel('Component'); ylabel('Variance explained');
title('Scree plot');
subplot(1,2,2);
plot(cumVarExplained, 'o-'); 
%%yline(targetVar, 'r--');
xlabel('Number of components'); ylabel('Cumulative variance explained');
title('Cumulative variance');


%{
 % Project training faces into 3D eigenface space
figure('Name', '3D Eigenfaces Space');
dim_reduced = 3;

% Extract first 3 principal components for each face
coords_3d = weights(1:dim_reduced, :);

% Create 3D scatter plot
scatter3(coords_3d(1,:), coords_3d(2,:), coords_3d(3,:), 50, 'filled');
xlabel(sprintf('PC 1 (%.1f%%)', 100*varExplained(1)));
ylabel(sprintf('PC 2 (%.1f%%)', 100*varExplained(2)));
zlabel(sprintf('PC 3 (%.1f%%)', 100*varExplained(3)));
title('Training Faces in 3D Eigenfaces Space');
grid on; 
%}


% filepath: /Users/aakim/ProjectsMatlab/SingularValueDecomposition/EigenFaces/dev/Eigenfaces_RaceEffects.m

