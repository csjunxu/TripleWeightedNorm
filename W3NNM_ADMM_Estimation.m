function [ EPat, WPat, ErrorRow ] = W3NNM_ADMM_Estimation( Y, NY, NL_mat, SigmaRow, SigmaCol, Par )

EPat  = zeros(size(Y));
WPat = zeros(size(Y));
ErrorRow = zeros(size(Y, 1), length(Par.SelfIndex));
for  i      =  1 : length(Par.SelfIndex) % For each keypatch group
    Temp    =   Y(:, NL_mat(1:Par.nlsp,i)); % Non-local similar patches to the keypatch
    M_Temp  =   repmat(mean( Temp, 2 ),1,Par.nlsp);
    Temp    =   Temp-M_Temp;
    E_Temp 	=   W3NNM_ADMM( Temp, SigmaRow(:, i), SigmaCol(:, NL_mat(1:Par.nlsp,i)), Par); % WNNM Estimation
    E_Temp = E_Temp + M_Temp;
    % update ErrorRow
    ErrorRow(:, i) = mean((NY(:, NL_mat(1:Par.nlsp,i)) - E_Temp) .^2, 2);
    % 
    EPat(:,NL_mat(1:Par.nlsp,i))  = EPat(:,NL_mat(1:Par.nlsp,i))+E_Temp;
    WPat(:,NL_mat(1:Par.nlsp,i))     = WPat(:,NL_mat(1:Par.nlsp,i))+ones(Par.ps2ch, Par.nlsp);
end
end

