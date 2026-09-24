% Generate ground-truth grape overlays and dataset plots without modifying source data.
scriptPath = mfilename('fullpath');
projectRoot = fileparts(fileparts(scriptPath));
annotationFile = fullfile(projectRoot, 'VINEPICs', 'data', 'annotations', 'VINEPICs_annotations.json');
imageRoot = fullfile(projectRoot, 'VINEPICs', 'data', 'images');
outputRoot = fullfile(projectRoot, 'results', 'dataset_checks');
if ~exist(outputRoot, 'dir'), mkdir(outputRoot); end

data = jsondecode(fileread(annotationFile));
images = data.images;
annotations = data.annotations;
imageIds = [images.id];
annotationImageIds = [annotations.image_id];
clustersPerImage = arrayfun(@(id) sum(annotationImageIds == id), imageIds);

% Map image IDs to dimensions for normalized bounding-box areas.
dimensions = containers.Map('KeyType', 'double', 'ValueType', 'any');
for index = 1:numel(images)
    dimensions(images(index).id) = [double(images(index).width), double(images(index).height)];
end
boxAreas = zeros(1, numel(annotations));
for index = 1:numel(annotations)
    dims = dimensions(annotations(index).image_id);
    box = double(annotations(index).bbox);
    boxAreas(index) = 100.0 * box(3) * box(4) / (dims(1) * dims(2));
end

% Plot distribution of cluster counts and relative box areas.
fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1300 500]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile;
histogram(clustersPerImage, 'BinMethod', 'integers', 'FaceColor', [0.44 0.18 0.66]);
xlabel('Annotated grape clusters'); ylabel('Images'); title('Grape clusters per image'); grid on;
nexttile;
histogram(boxAreas, 30, 'FaceColor', [0.25 0.56 0.24]);
xlabel('Bounding-box area (% of image)'); ylabel('Boxes'); title('Cluster box-size distribution'); grid on;
exportgraphics(fig, fullfile(outputRoot, 'dataset_distributions.png'), 'Resolution', 180);
close(fig);

% Plot image and cluster counts by capture session.
fileNames = string({images.file_name});
sessions = extractBefore(fileNames, '/');
[sessionNames, ~, sessionIndex] = unique(sessions, 'stable');
sessionImageCounts = accumarray(sessionIndex(:), 1);
sessionClusterCounts = zeros(numel(sessionNames), 1);
for index = 1:numel(sessionNames)
    ids = imageIds(sessionIndex == index);
    sessionClusterCounts(index) = sum(ismember(annotationImageIds, ids));
end
fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1500 650]);
yyaxis left; bar(sessionImageCounts, 0.75, 'FaceColor', [0.22 0.49 0.72]); ylabel('Images');
yyaxis right; plot(1:numel(sessionNames), sessionClusterCounts, '-o', 'LineWidth', 2.2, ...
    'Color', [0.75 0.22 0.17], 'MarkerFaceColor', [0.75 0.22 0.17]); ylabel('Annotated clusters');
xticks(1:numel(sessionNames)); xticklabels(sessionNames); xtickangle(45);
xlabel('Capture session'); title('Dataset coverage by capture session'); grid on;
exportgraphics(fig, fullfile(outputRoot, 'clusters_by_session.png'), 'Resolution', 180);
close(fig);

% Representative Red Globe, Ortrugo, and Cabernet Sauvignon images.
selected = [
    "2021-08-23/rgb27.png"
    "2022-08-23-15-32-40/rgb-2022-08-23-15-34-03-802105.png"
    "2022-09-15-14-08-23/rgb-2022-09-15-14-08-44-200591.png"
];
variety = ["Red Globe", "Ortrugo", "Cabernet Sauvignon"];
fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1500 850]);
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
for panel = 1:numel(selected)
    imageIndex = find(fileNames == selected(panel), 1);
    if isempty(imageIndex), continue; end
    imageInfo = images(imageIndex);
    rgb = imread(fullfile(imageRoot, strrep(char(selected(panel)), '/', filesep)));
    nexttile; imshow(rgb); hold on;
    matches = find(annotationImageIds == imageInfo.id);
    for boxIndex = 1:numel(matches)
        box = double(annotations(matches(boxIndex)).bbox);
        rectangle('Position', box, 'EdgeColor', [0.1 1.0 0.1], 'LineWidth', 2.2);
        text(box(1), max(1, box(2) - 4), sprintf('G%03d', boxIndex), ...
            'Color', 'white', 'FontWeight', 'bold', 'FontSize', 8, ...
            'BackgroundColor', [0.05 0.55 0.05], 'Margin', 1);
    end
    title(sprintf('%s - %d clusters', variety(panel), numel(matches)), 'Interpreter', 'none');
    hold off;
end
sgtitle('VINEPICs ground-truth grape-cluster annotations');
exportgraphics(fig, fullfile(outputRoot, 'annotated_grape_examples.png'), 'Resolution', 180);
close(fig);

fprintf('Created annotation plots in %s\n', outputRoot);
