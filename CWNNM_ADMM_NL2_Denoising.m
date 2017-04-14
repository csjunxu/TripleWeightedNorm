function [E_Img, Par]   =  CWNNM_ADMM_NL2_Denoising( N_Img, O_Img, Par )
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
Sigma = ones(Par.ch, length(Par.SelfIndex));
for iter = 1 : Par.Iter
    Par.iter = iter;
    % iterative regularization
    E_Img =	E_Img + Par.delta * (N_Img - E_Img);
    % image to patch
    CurPat =	Image2PatchNew( E_Img, Par );
    %     CurPatCh = [CurPat(1:Par.ps2, :) CurPat(Par.ps2+1:2*Par.ps2, :) CurPat(2*Par.ps2+1:3*Par.ps2, :)];
    % estimate local noise variance
    for c = 1:Par.ch
        if(iter == 1)
            TempSigma_arrCh = Par.lambda(Par.iter) * Par.nSig0(c) * Sigma(c, :);
            Sigma_arrCh((c-1)*Par.ps2+1:c*Par.ps2, :) = repmat(TempSigma_arrCh, [Par.ps2, 1]);
        else
            Sigma_arrCh = Par.lambda(Par.iter) * Sigma;
        end
    end
    
    if (mod(iter-1, Par.Innerloop) == 0)
        Par.nlsp = Par.nlsp - 10;  % Lower Noise level, less NL patches
        NL_mat  =  Block_Matching(CurPat, Par);% Caculate Non-local similar patches for each
    end
    % Inexact ALM for Weighted WNNM
    [Y_hat, W_hat, Sigma]  =  CWNNM_ADMM_NL2_Estimation( NL_mat, Sigma_arrCh, NoiPat, Par );   % Estimate all the patches
    E_Img = PGs2Image(Y_hat, W_hat, Par);
    PSNR  = csnr( O_Img, E_Img, 0, 0 );
    SSIM      =  cal_ssim( O_Img, E_Img, 0, 0 );
    fprintf( 'Iter = %2.3f, PSNR = %2.2f, SSIM = %2.2f \n', iter, PSNR, SSIM );
    Par.PSNR(iter, Par.image)  =   PSNR;
    Par.SSIM(iter, Par.image)      =  SSIM;
end
return;





