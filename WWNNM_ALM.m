
function  Z =  WWNNM_ALM( Y, NSig, Par )
% This routine solves the following weighted nuclear norm optimization problem with column weights,
%
% min |Z|_*,P + |Y-XW|_2,1  s.t.,  X = Z
% inputs:
%        Y -- D*N data matrix, D is the data dimension, and N is the number
%             of image patches.
%        W -- N*N matrix of column weights

% tol = 1e-8;
% max_mu = 1e10;
if ~isfield(Par, 'maxIter')
    Par.maxIter = 10;
end
if ~isfield(Par, 'rho')
    Par.rho = 1;
end
if ~isfield(Par, 'display')
    Par.display = true;
end
% if ~isfield(Par, 'mu')
%     Par.mu = 1.1;
% end
%% Initializing optimization variables
% intialize

if strcmp(Par.method, 'WNNM') ==1
    W = ones(1, length(NSig));
else
    W = NSig(1) ./ NSig;
end
X = zeros(size(Y));
Z = zeros(size(Y));
A = zeros(size(Y));
%% Start main loop
iter = 0;
PatNum       = size(Y,2);
TempC  = Par.Constant * sqrt(PatNum) * NSig(1)^2;
while iter < Par.maxIter
    iter = iter + 1;
    % update X, fix Z and A
    % min_{X} ||Y - XW||_F^2 + 0.5 * mu * ||X - (Z - 1/mu * A)||_F^2
    X = (Y * diag(W.^2) + 0.5 * Par.rho * Z - 0.5 * A) * diag(1 ./ (W.^2 + 0.5 * Par.rho));
    % update Z, fix X and A
    % min_{Z} |Z|_*,P + 0.5 * mu * |Z - (X + 1/mu * A)|_F^2
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
    %         % update the multiplier A, fix Z and X
    A = A + Par.rho * (X - Z);
%     % update the parameter mu
%     Par.rho = min(max_mu, Par.mu * Par.rho);
    %     end
end
return;
