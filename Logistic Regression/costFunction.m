function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


s = (-y' * log(sigmoid(X*theta)) - (1-y)' *  log(1-sigmoid(X*theta)));

J = (1/m) * (s);

grad = (1/m) * (sigmoid(X*theta)-y)' * X;


% =============================================================

end
