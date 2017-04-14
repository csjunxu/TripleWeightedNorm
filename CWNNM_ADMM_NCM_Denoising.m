function [E_Img, Par]   =  CWNNM_ADMM_NCM_Denoising( N_Img, O_Img, Par )
E_Img           = N_Img;   % Estimated Image
[h, w, ch]  = size(E_Img);
Par.h = h;
Par.w = w;
Par.ch = ch;
Par = SearchNeighborIndex( Par );
% noisy image to patch
NoiPat =	Image2PatchNew( N_Img, Par );
% NoiPatCh = [NoiPat(1:Par.ps2, :) NoiPat(Par.ps2+1:2*Par.ps2, :) NoiPat(2*Par.ps2+1:3*Par.ps2, :)];
Par.TolN = size(NoiPat, 2);
for iter = 1 : Par.Iter
    Par.iter = iter;
    % iterative regularization
    E_Img =	E_Img + Par.delta * (N_Img - E_Img);
    % image to patch
    CurPat =	Image2PatchNew( E_Img, Par );
    %     CurPatCh = [CurPat(1:Par.ps2, :) CurPat(Par.ps2+1:2*Par.ps2, :) CurPat(2*Par.ps2+1:3*Par.ps2, :)];
    % estimate local noise variance
    for c = 1:Par.ch
        TempSigma_arrCh = Par.lambda(Par.iter) * Par.nSig0(c);
        Sigma_arrCh((c-1)*Par.ps2+1:c*Par.ps2, 1) = repmat(TempSigma_arrCh, [Par.ps2, 1]);
    end
    if (mod(iter-1, Par.Innerloop) == 0)
%         Par.nlsp = Par.nlsp - 10;  % Lower Noise level, less NL patches
        NL_mat  =  Block_Matching(CurPat, Par);% Caculate Non-local similar patches for each
    end
    % setting the weight matrix W
    if (iter==1)
        if strcmp(Par.method, 'WNNM_ADMM') ==1
            sigma = sqrt(mean(Sigma_arrCh.^2)) + eps;
            W = 1/sigma * eye(Par.ps2ch);
        else
            W = diag(1 ./ (Sigma_arrCh+eps));
        end
        W = repmat({W}, [length(Par.SelfIndex), 1]);
    else
        weights = diag(sqrt(1./Sigma_arrCh));
        if strcmp(Par.method, 'WNNM_ADMM') ==1
            W = weights * eye(Par.ps2ch) * weights;
            W = repmat({W}, [length(Par.SelfIndex), 1]);
        else
            W = cellfun(@(a,b) weights*a*weights, WCM, 'UniformOutput', 0);
        end
    end
    % Inexact ALM for Weighted WNNM
    [Y_hat, W_hat, WCM]  =  CWNNM_ADMM_NCM_Estimation( NL_mat, W, NoiPat, Par );   % Estimate all the patches
    E_Img = PGs2Image(Y_hat, W_hat, Par);
    PSNR  = csnr( O_Img, E_Img, 0, 0 );
    SSIM      =  cal_ssim( O_Img, E_Img, 0, 0 );
    fprintf( 'Iter = %2.3f, PSNR = %2.2f, SSIM = %2.2f \n', iter, PSNR, SSIM );
    Par.PSNR(iter, Par.image)  =   PSNR;
    Par.SSIM(iter, Par.image)      =  SSIM;
end
return;





