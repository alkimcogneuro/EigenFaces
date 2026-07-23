function [targetFaces] = read_face_data(weights)  
    
    % Project training faces into 3D eigenface space
    figure('Name', '3D Eigenfaces Space');
    dim_reduced = 3;
    
    % Extract first 3 principal components for each face
    coords_3d = weights(1:dim_reduced, :);
    
    % Create 3D scatter plot
    scatter3(coords_3d(1,:), coords_3d(2,:), coords_3d(3,:), 50, 'filled');
    hold on;
    
    % Draw vector lines from origin to each point
    origin = [0; 0; 0];
    for i = 1:size(coords_3d, 2)
        plot3([origin(1), coords_3d(1,i)], [origin(2), coords_3d(2,i)], [origin(3), coords_3d(3,i)], 'k-', 'LineWidth', 0.5);
    end
    
    hold off;
    xlabel(sprintf('PC 1 (%.1f%%)', 100*varExplained(1)));
    ylabel(sprintf('PC 2 (%.1f%%)', 100*varExplained(2)));
    zlabel(sprintf('PC 3 (%.1f%%)', 100*varExplained(3)));
    title('Training Faces in 3D Eigenfaces Space');
    grid on;
end
