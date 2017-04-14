function  [Z] =  CWNNM_ADMM1( Y, NSig, Par )
% This routine solves the following weighted nuclear norm optimization problem with column weights,
%
% min |Z|_*,P + |Y-XW|_2,1  s.t.,  X = Z
% inputs:
%        Y -- D*N data matrix, D is the data dimension, and N is the number
%             of image patches.
%        W -- N*N matrix of column weights

% tol = 1e-8;
maxrho = 100;
if ~isfield(Par, 'maxIter')
    Par.maxIter = 10;
end
if ~isfield(Par, 'rho')
    Par.rho = 1;
end
if ~isfield(Par, 'mu')
    Par.mu = 1;
end
if ~isfield(Par, 'display')
    Par.display = true;
end
% Initializing optimization variables
% Intialize the weight matrix W
mNSig = min(NSig);
if strcmp(Par.method, 'WNNM_ADMM') ==1
    W = ones(1, length(NSig));
else
    W = (mNSig+eps) ./ (NSig+eps);
end
% Initializing optimization variables
X = zeros(size(Y));
Z = zeros(size(Y));
A = zeros(size(Y));
%% Start main loop
iter = 0;
PatNum       = size(Y,2);
TempC  = Par.Constant * sqrt(PatNum) * mNSig^2;
Par.rho = Par.rho * (mNSig+eps)^2;

while iter < Par.maxIter
    iter = iter + 1;
    
    % update X, fix Z and A
    % min_{X} ||W * Y - W * X||_F^2 + 0.5 * rho * ||X - Z + 1/rho * A||_F^2
    X = diag(1 ./ (W.^2 + 0.5 * Par.rho)) * (diag(W.^2) * Y + 0.5 * Par.rho * Z - 0.5 * A);
    
    % update Z, fix X and A
    % min_{Z} ||Z||_*,w + 0.5 * rho * ||Z - (X + 1/rho * A)||_F^2
    Temp = X + A/Par.rho;
    [U, SigmaTemp, V] =   svd(full(Temp), 'econ');
    [SigmaZ, svp] = ClosedWNNM(diag(SigmaTemp), 2/Par.rho*TempC, eps);
    Z =  U(:, 1:svp) * diag(SigmaZ) * V(:, 1:svp)';
    %     % check the convergence conditions
    %     stopC = max(max(abs(X - Z)));
    %     if Par.display && (iter==1 || mod(iter,10)==0 || stopC<tol)
    %         disp(['iter ' num2str(iter) ',mu=' num2str(Par.mu,'%2.1e') ...
    %             ',rank=' num2str(rank(Z,1e-4*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    %     end
    %     if stopC < tol
    %         break;
    %     else
    % update the multiplier A, fix Z and X
    A = A + Par.rho * (X - Z);
    Par.rho = min(maxrho, Par.mu * Par.rho);
    %     end
end
return;
