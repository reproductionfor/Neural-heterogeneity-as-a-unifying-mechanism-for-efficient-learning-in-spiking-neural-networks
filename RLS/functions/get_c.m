function C = get_c(X, alpha)
    % get_c - Regularized covariance-like matrix
    %
    % Syntax: C = get_c(X, alpha)
    %
    % Inputs:
    %   X     - [D x N] input matrix
    %   alpha - scalar regularization parameter (optional, default = 1e-4)
    %
    % Output:
    %   C     - regularized matrix: X*X' + alpha*I

    if nargin < 2
        alpha = 1e-4;
    end

    C = X * X' + alpha * eye(size(X, 1));
end