function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================
m = size(y,1);		% Size of the vector
pos = find(y==1);	% Find all positive values in y
neg = find(y==0);	% Find all negative values in y

plot(X(pos, 1), X(pos, 2),		%Way to find a sub-matrix selection
	'k+', 'LineWidth', 2, 'MarkerSize',7, 'Color', 'red');
plot(X(neg, 1), X(neg, 2),
	'ko', 'LineWidth', 2, 'MarkerSize', 7, 'Color', 'green');

hold off;

end
