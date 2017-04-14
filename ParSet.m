function  [Par]=ParSet(nSig)

Par.nSig      =   nSig;                                 % Variance of the noise image
Par.win =   30;                                   % Non-local patch searching window
Par.delta     =   0.1;                                  % Parameter between each iter
Par.Constant         =   2 * sqrt(2);                              % Constant num for the weight vector
Par.Innerloop =   2;                                    % InnerLoop Num of between re-blockmatching
Par.ReWeiIter =   3;
if nSig<=20
    Par.ps       =   6;                            % Patch size
    Par.nlsp        =   70;                           % Initial Non-local Patch number
    Par.Iter          =   8;                            % total iter numbers
    Par.lamada        =   0.54;                         % Noise estimete parameter
elseif nSig <= 40
    Par.ps       =   7;
    Par.nlsp        =   90;
    Par.Iter          =   12;
    Par.lamada        =   0.56; 
elseif nSig<=60
    Par.ps       =   8;
    Par.nlsp        =   120;
    Par.Iter          =   14;
    Par.lamada        =   0.58; 
else
    Par.ps       =   9;
    Par.nlsp        =   140;
    Par.Iter          =   14;
    Par.lamada        =   0.58; 
end
Par.step      =   floor(Par.ps-1);      
% par.step      =   floor((par.ps)/2-1);                   
% Blockmatching and perform WNNM algorithm on all the patches in the image
% is time consuming, we just perform the blockmatching and WNNM on parts of
% patches in the image (we call these patches keypatch in explanatory notes)
% par.step is the step between each keypatch, smaller step will further
% improve the denoisng result