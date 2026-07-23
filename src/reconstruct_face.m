function [face_reconstructed] = reconstruct_face(face_space_basis,  allfaces_meancentered, num_dims, face_idx, mean_face)  
    if isequal(num_dims, 'all')
        num_dims = size(face_space_basis, 2);  % Use all eigenfaces if "all" is specified
    end 

    face_weights_all = face_space_basis' * allfaces_meancentered;  % project all faces into the eigenface space
    face_weights_target = face_weights_all(: , face_idx);               % grab the weights for the target face
    % Use just the first num_dims weights and first num_dims eigenfaces to reconstruct the target face
    % If num_dims is equal to the number of eigenfaces, the reconstruction will be exact
    face_reconstructed = mean_face + (face_space_basis(:,1:num_dims) * face_weights_target(1:num_dims));  
    % ------------------------------------------------------------------------------------
    % Reconstruct one face from the Eigenface Weights
    % Any mean-centered face can be reconstructed exactly (using all components) as: 
    % A(:,i) = U * face_weights
    % The face_weights vector holds the coordinates the face in the eigenface basis, 
    % while U holds the eigenface basis vectors themselves (coordinates of those vectors in the original pixel space).
    % The matrix multiplication U * face_weights is a linear combination of the eigenfaces (columns of U) 
    % weighted by the coordinates of face in eigenface space (face_weights).
    % This provides the coordinates of the target face in the original pixel space.
    
    % The faces have been mean-centered, so the reconstructed faces are also mean-centered.
    % If we want to reconstruct the original, uncentered face, we add the mean face back in:
    % reconstructed_faces(:,i) = meanFace + U*face_weights
    
    % ------------------------------------------------------------------------------------
    %% face_reconstructed  = meanFace + (face_space_basis * face_weights);
end
