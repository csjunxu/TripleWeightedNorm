function  [Z] =  W3NNM_ADMM( Y, NSigRow, NSigCol, Par )
% This routine solves the following weighted nuclear norm optimization problem with column weights,
%
% min |Z|_*,P + |W1(Y-X)W2|_2,1  s.t.,  X = Z
% inputs:
%        Y -- d*M data matrix, d is the data dimension, and M is the number
%             of image patches.
%        W1 -- d*d matrix of row weights
%        W2 -- M*M matrix of column weights

% tol = 1e-12; 
% Initializing optimization variables
% Intialize the weight matrix W
if strcmp(Par.method, 'WNNM_ADMM') ==1
    sigma = sqrt(mean(NSigRow.^2)) + eps;
    W1 = 1/sigma * ones(1, size(NSigRow, 1));
    W2 = 1/NSigCol(1, 1) * ones(1, size(NSigCol, 2));
else
    W1 = 1 ./ (NSigRow(:, 1)+eps);
    W2 = 1 ./ (NSigCol(1, :)+eps);
end
% Initializing optimization variables
X = zeros(size(Y));
Z = zeros(size(Y));
D = zeros(size(Y));
%% Start main loop
iter = 0;
PatNum       = size(Y,2);
TempC  = Par.Constant * sqrt(PatNum);
while iter < Par.maxIter
    iter = iter + 1;
    
    % update X, fix Z and D
    % min_{X} ||W1 * (Y - X) * W2||_F^2 + 0.5 * rho * ||X - Z + 1/rho * A||_F^2
    % The solution is equal to solve A * X + X * B =C
    A = diag(W1.^2);
    B = 0.5 * Par.rho * diag(1./(W2.^2));
    C = diag(W1.^2) * Y + ( 0.5 * Par.rho * Z - 0.5 * D ) * diag(1./(W2.^2));
    X = sylvester(A, B, C);
    
    % update Z, fix X and A
    % min_{Z} ||Z||_*,w + 0.5 * rho * ||Z - (X + 1/rho * D)||_F^2
    Temp = X + D/Par.rho;
    [U, SigmaTemp, V] =   svd(full(Temp), 'econ');
    [SigmaZ, svp] = ClosedWNNM(diag(SigmaTemp), 2/Par.rho*TempC, eps);
    Z =  U(:, 1:svp) * diag(SigmaZ) * V(:, 1:svp)';
    %     % check the convergence conditions
    %     stopC = max(max(abs(X - Z)));
    %     if Par.display && (iter==1 || mod(iter,10)==0 || stopC<tol)
    %         disp(['iter ' num2str(iter) ', mu=' num2str(Par.mu,'%2.1e') ...
    %             ', rank=' num2str(rank(Z,1e-4*norm(Z,2))) ', stopADMM=' num2str(stopC,'%2.3e')]);
    %     end
    %     if stopC < tol
    %         break;
    %     else
    % update the multiplier A, fix Z and X
    D = D + Par.rho * (X - Z);
    Par.rho = min(Par.maxrho, Par.mu * Par.rho);
    %     end
end
return;
