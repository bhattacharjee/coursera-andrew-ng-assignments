function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


m = size(y, 1);
H_X = sigmoid(X * theta);

cost_regularization_penalty = lambda * (theta' * theta - theta(1) * theta(1)) / (2 * m);
J = (-1 / m) * ((y' * log(H_X)) + ((1 .- y)' * log(1 .- H_X))) + cost_regularization_penalty;

grad = (((H_X - y)' * X)') ./ m;
grad_add_regularization = theta * (lambda / m);
grad_add_regularization(1) = 0;
grad = grad + grad_add_regularization;

% =============================================================

end
