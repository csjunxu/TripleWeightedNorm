
function  [X] =  WNNM( Y, C, NSig, m, Iter )
[U,SigmaY,V] =   svd(full(Y),'econ');
PatNum       = size(Y,2);
TempC  = C*sqrt(PatNum)*NSig^2;
[SigmaX,svp] = ClosedWNNM(diag(SigmaY),TempC,eps);
X = U(:,1:svp)*diag(SigmaX)*V(:,1:svp)' + m;

return;
