function [ EPat, WPat, WCM] = CWNNM_ADMM_NCM_Estimation( NL_mat, W, CurPat, Par )

EPat  = zeros(size(CurPat));
WPat = zeros(size(CurPat));
WCM = cell(length(Par.SelfIndex), 1);
for  i      =  1 : length(Par.SelfIndex) % For each keypatch group
    Temp    =   CurPat(:, NL_mat(1:Par.nlsp,i)); % Non-local similar patches to the keypatch
    M_Temp  =   repmat(mean( Temp, 2 ),1,Par.nlsp);
    Temp    =   Temp-M_Temp;
    [E_Temp, WCM{i}] 	=   CWNNM_ADMM_NCM( Temp, W{i}, Par); % WNNM Estimation
    EPat(:,NL_mat(1:Par.nlsp,i))  = EPat(:,NL_mat(1:Par.nlsp,i))+E_Temp+M_Temp;
    WPat(:,NL_mat(1:Par.nlsp,i))     = WPat(:,NL_mat(1:Par.nlsp,i))+ones(Par.ps2ch, Par.nlsp);
end
end

