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

for i=1:m
  h_theta_X = sigmoid(X(i,:) * theta);
  J = J + (-1*y(i)*log(h_theta_X) - (1-y(i))*log(1-h_theta_X));
endfor
J = J / m + lambda / (2*m) * (theta(2:size(theta))' * theta(2:size(theta)));
 
for i=1:m
  h_theta_X = sigmoid(X(i,:) * theta);
  grad = grad + (h_theta_X - y(i)) * X(i,:)'; 
endfor
grad = grad / m + lambda / m * [0; theta(2:size(theta))];


% =============================================================

end
