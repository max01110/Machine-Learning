function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));



%Cost:
predRatings = X*Theta';
error = predRatings-Y;
unRegJ = (1/2)*sum((error.^2)(R==1));

%Cost - Regularization:
regJ = (lambda/2) * sum(sum(Theta.*Theta));
regJ = regJ + (lambda/2) * sum(sum(X.*X));

J = unRegJ + regJ;


%Gradient:
errorFactor = error.*R;
X_grad_UnReg = errorFactor * Theta;
Theta_grad_UnReg = errorFactor' * X;

%Gradient - Regularization:

X_grad_Reg = lambda * X;
Theta_grad_Reg = lambda * Theta;


X_grad = X_grad_UnReg + X_grad_Reg;
Theta_grad = Theta_grad_UnReg + Theta_grad_Reg;




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
