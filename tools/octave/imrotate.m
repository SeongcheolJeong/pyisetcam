function out = imrotate(im, angle)
% Minimal imrotate shim for curated Codex parity cases.

if abs(angle) < eps
    out = im;
    return;
end

[rows, cols] = size(im);
cx = (cols + 1) / 2;
cy = (rows + 1) / 2;
theta = angle * pi / 180;

corners = [
    1, 1;
    cols, 1;
    cols, rows;
    1, rows
];

rot = [
    cos(theta), -sin(theta);
    sin(theta),  cos(theta)
];

centered = corners - [cx, cy];
rotated = centered * rot';
rotated = rotated + [cx, cy];

xMin = floor(min(rotated(:, 1)));
xMax = ceil(max(rotated(:, 1)));
yMin = floor(min(rotated(:, 2)));
yMax = ceil(max(rotated(:, 2)));

[xOut, yOut] = meshgrid(xMin:xMax, yMin:yMax);
xShift = xOut - cx;
yShift = yOut - cy;

xIn =  cos(theta) * xShift + sin(theta) * yShift + cx;
yIn = -sin(theta) * xShift + cos(theta) * yShift + cy;

out = interp2(1:cols, 1:rows, double(im), xIn, yIn, 'linear', 0);
if isa(im, 'single')
    out = single(out);
end
end
