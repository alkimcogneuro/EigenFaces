a
    % grab a subset of faces based on the specified start and end columns
    % the CSV file is assumed to have a header row, which will be ignored.
    % the CSV file contains many different faces, 
    % and the user can specify which subset of faces to read in by providing the start and end column indices.

    AllFaces  = readtable(filename);
    headers = AllFaces.Properties.VariableNames;
    AllFacesMatrix = table2array(AllFaces);                 % numeric matrix, headers stored separately
    %-----------------------------------------------------
    targetFaces = AllFacesMatrix(:,startcol:endcol);        % grab a subet of faces based on the specified start and end columns
end
