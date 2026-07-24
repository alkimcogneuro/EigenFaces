function [] = plot_all_faces(faces_all, num_rows_face, num_cols_face)  
    % Plot all training faces in a grid layout
    nFaces = size(faces_all, 2);
    figure('Name', 'All Training Faces');
    nColsGrid = ceil(sqrt(nFaces));
    nRowsGrid = ceil(nFaces / nColsGrid);
    
    for faceIdx = 1:nFaces
        subplot(nRowsGrid, nColsGrid, faceIdx);
        face_img = reshape(faces_all(:, faceIdx), num_rows_face, num_cols_face);
        imagesc(face_img);
        axis image off;
        colormap gray;
    end
end
