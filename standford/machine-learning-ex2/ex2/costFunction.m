function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

for i=1:m
  h_theta_X = sigmoid(X(i,:) * theta);
  J = J + (-1*y(i)*log(h_theta_X) - (1-y(i))*log(1-h_theta_X));
endfor
J = J / m;
 
for i=1:m
  h_theta_X = sigmoid(X(i,:) * theta);
  grad = grad + (h_theta_X - y(i)) * X(i,:)'; 
endfor
grad = grad / m;


% =============================================================

end
