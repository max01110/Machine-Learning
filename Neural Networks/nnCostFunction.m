function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

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



y_matrix = eye(num_labels)(y,:);

a1 = [ones(size(X, 1), 1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
y1 = eye(num_labels)(y, :);
cost = ( -y1 .* log(a3) ) -(1-y1) .* log(1-a3); 
J = sum(cost(:));
J = (1/m) * J;

Theta1_reg = Theta1(:, 2:end);
Theta2_reg = Theta2(:, 2:end);

regularization = (lambda/(2*m)) * ( sum( (Theta1_reg.^2)(:) ) + sum( (Theta2_reg.^2)(:) ) );
J = J + regularization;
%-------Gradients-------------


d3 = a3-y_matrix;
z2Grad = sigmoidGradient(z2);
d2 = (d3*Theta2_reg) .* z2Grad;
Delta1 = d2' * a1;
Delta2 = d3' * a2;
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

%Regularization

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1 = Theta1 * (lambda/m);
Theta2 = Theta2 * (lambda/m);

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
