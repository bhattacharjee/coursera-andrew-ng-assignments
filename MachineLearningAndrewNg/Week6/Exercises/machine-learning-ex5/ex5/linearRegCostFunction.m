function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% X = 12 * 2
% Theta = 2 * 1
% y = 12 * 1


H = X * theta;	% 12 * 1
H_minus_y = H - y;

J = (1/(2*m)) * sum(H_minus_y .* H_minus_y);
regCost =  (lambda / (2 * m)) * (theta' * theta - theta(1)' * theta(1));
J = J + regCost;

% =========================================================================

grad = (1/m) * X' * H_minus_y;
theta_dash = theta(2:end,:);
theta_dash = [zeros(1, size(theta_dash, 2)); theta_dash];
grad = grad + (lambda / m) * theta_dash;

end
