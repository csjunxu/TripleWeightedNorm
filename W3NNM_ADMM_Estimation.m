function [ EPat, WPat ] = W3NNM_ADMM_Estimation( Y, NL_mat, SigmaRow, SigmaCol, Par )

EPat  = zeros(size(Y));
WPat = zeros(size(Y));
for  i      =  1 : length(Par.SelfIndex) % For each keypatch group
    Temp    =   Y(:, NL_mat(1:Par.nlsp,i)); % Non-local similar patches to the keypatch
    M_Temp  =   repmat(mean( Temp, 2 ),1,Par.nlsp);
    Temp    =   Temp-M_Temp;
    E_Temp 	=   W3NNM_ADMM( Temp, SigmaRow(:, Par.SelfIndex(i)), SigmaCol(:, NL_mat(1:Par.nlsp,i)), Par); % WNNM Estimation
    % update W1
    W1(index) = exp( - par.lambdals * sqrt(sum((nDCnlY - nDCnlYhat) .^2, 1)) );
    EPat(:,NL_mat(1:Par.nlsp,i))  = EPat(:,NL_mat(1:Par.nlsp,i))+E_Temp+M_Temp;
    WPat(:,NL_mat(1:Par.nlsp,i))     = WPat(:,NL_mat(1:Par.nlsp,i))+ones(Par.ps2ch, Par.nlsp);
end
end

