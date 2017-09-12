function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Feed Forward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h1 = sigmoid([ones(m, 1) X] * Theta1');
H = sigmoid([ones(m, 1) h1] * Theta2');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% COST FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[t1m, t1n] = size(Theta1);
%[t2m, t2n] = size(Theta2);

% Compute the regularization cost
% instead of using a loop (to avoid the last column),
% just add everything, and then subtract the last column
theta_temp = Theta1(:,1);
regularizationCost = sum(sum(Theta1 .* Theta1)) - ...
			sum(sum(theta_temp .* theta_temp));
theta_temp = Theta2(:,1);
regularizationCost = regularizationCost + sum(sum(Theta2 .* Theta2)) - ...
			sum(sum(theta_temp .* theta_temp));
regularizationCost = (lambda / (2 * m)) * regularizationCost;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% CONVERT y FROM A NUMBER TO n OUTPUT LINES %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yy = zeros(m, num_labels);
for i = 1:m
    yy(i, y(i)) = 1;
end

J = -yy .* log(H) - (1 .- yy) .* log(1 .- H);
J = 1/m * sum(sum(J));
J = J + regularizationCost;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% NOW TO COMPUTE THE GRADIENT %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Theta1				  25 x 401
% Theta2				  10 x 26
a1 = [ones(size(X,1),1) X];		% 5000 x 401
z2 = a1 * Theta1';			% 5000 x 25
a2 = [ones(size(z2,1),1) sigmoid(z2)];	% 5000 x 26
z3 = a2 * Theta2';			% 5000 x 10
a3 = sigmoid(z3);			% 5000 x 10

d3 = a3 - yy;				% 5000 x 10
d2 = (d3 * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)];	% 5000 x 26

theta1 = Theta1(:,2:size(Theta1,2));%25*400
theta1 = [zeros(size(theta1,1),1),theta1];
Theta1_grad = (a1' * d2)';				% 26 x 401
Theta1_grad = 1/m * Theta1_grad(2:end, :);
Theta1_grad = Theta1_grad + (lambda / m) * theta1;

theta2 = Theta2(:,2:size(Theta2,2)); %10*25
theta2 = [zeros(size(theta2,1),1),theta2];
Theta2_grad = 1/m * (a2' * d3)';			% 10 x 26
Theta2_grad = Theta2_grad + (lambda / m) * theta2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];
