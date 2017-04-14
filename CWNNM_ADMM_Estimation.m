function [ EPat, W] = CWNNM_ADMM_Estimation( NL_mat, Sigma_arr, CurPat, Par )

EPat = zeros(size(CurPat));
W    = zeros(size(CurPat));
for  i      =  1 : length(Par.SelfIndex) % For each keypatch group
    Temp    =   CurPat(:, NL_mat(1:Par.nlsp,i)); % Non-local similar patches to the keypatch
    M_Temp  =   repmat(mean( Temp, 2 ),1,Par.nlsp);
    Temp    =   Temp-M_Temp;
    E_Temp 	=   CWNNM_ADMM( Temp, Sigma_arr(:, Par.SelfIndex(i)), Par); % WNNM Estimation
    EPat(:,NL_mat(1:Par.nlsp,i))  = EPat(:,NL_mat(1:Par.nlsp,i))+E_Temp+M_Temp;
    W(:,NL_mat(1:Par.nlsp,i))     = W(:,NL_mat(1:Par.nlsp,i))+ones(Par.ps2ch, Par.nlsp);
end
end

