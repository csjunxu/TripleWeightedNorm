function [E_Img, Par]   =  W3NNM_ADMM_Denoising( N_Img, O_Img, Par )
E_Img           = N_Img;   % Estimated Image
[h, w, ch]  = size(E_Img);
Par.h = h;
Par.w = w;
Par.ch = ch;
Par = SearchNeighborIndex( Par );
% noisy image to patch
NY =	Image2PatchNew( N_Img, Par );
% NYCh = [NY(1:Par.ps2, :) NY(Par.ps2+1:2*Par.ps2, :) NY(2*Par.ps2+1:3*Par.ps2, :)];
Par.TolN = size(NY, 2);
SigmaRow = zeros(Par.ps2ch, length(Par.SelfIndex));
for ite = 1 : Par.Outerloop
    % iterative regularization
    E_Img =	E_Img + Par.delta * (N_Img - E_Img);
    % image to patch
    Y =	Image2PatchNew( E_Img, Par );
    %     YCh = [Y(1:Par.ps2, :) Y(Par.ps2+1:2*Par.ps2, :) Y(2*Par.ps2+1:3*Par.ps2, :)];
    % estimate local noise variance
    for c = 1:Par.ch
        if (ite == 1) && (Par.Outerloop > 1)
            TempSigmaRow = sqrt(repmat(Par.nSig(c)^2, Par.ps2, 1));
            SigmaRow((c-1)*Par.ps2+1:c*Par.ps2, :) = repmat(TempSigmaRow, [1, length(Par.SelfIndex)]);
            %             TempSigmaRow = sqrt(abs(repmat(Par.nSig(c)^2, 1, size(Y, 2)) - mean((NY((c-1)*Par.ps2+1:c*Par.ps2, :) - Y((c-1)*Par.ps2+1:c*Par.ps2, :)).^2)));
        else
            %             TempSigmaRow = Par.lambda1*sqrt(max(0, repmat(Par.nSig(c)^2, Par.ps2, length(Par.SelfIndex)) - ErrorRow((c-1)*Par.ps2+1:c*Par.ps2, :)));
            SigmaRow((c-1)*Par.ps2+1:c*Par.ps2, :) = exp( - Par.lambda1*sqrt(max(0, repmat(Par.nSig(c)^2, Par.ps2, length(Par.SelfIndex)) - ErrorRow((c-1)*Par.ps2+1:c*Par.ps2, :))));
            %             TempSigmaRow = Par.lambda1*sqrt(abs(repmat(Par.nSig(c)^2, 1, size(Y, 2)) - mean((NY((c-1)*Par.ps2+1:c*Par.ps2, :) - Y((c-1)*Par.ps2+1:c*Par.ps2, :)).^2)));
        end
    end
    SigmaCol = Par.lambda2*sqrt(abs(repmat(Par.nSig^2, 1, size(Y,2)) - mean((NY - Y).^2))); %Estimated Local Noise Level
    
    if (mod(ite-1, Par.Innerloop) == 0)
        Par.nlsp = Par.nlsp - 10;  % Lower Noise level, less NL patches
        NL_mat  =  Block_Matching(Y, Par);% Caculate Non-local similar patches for each
        if ite == 1 && Par.lambda1 ~= 0
            SigmaRow = ones(Par.ps2ch, length(Par.SelfIndex));
        end
        if ite == 1 && Par.lambda2~=0
            SigmaCol = Par.nSig * ones(size(SigmaCol));
        end
    end
    [Y_hat, W_hat, ErrorRow]  =  W3NNM_ADMM_Estimation( Y, NY, NL_mat, SigmaRow, SigmaCol, Par );   % Estimate all the patches
    E_Img = PGs2Image(Y_hat, W_hat, Par);
    PSNR  = csnr( O_Img, E_Img, 0, 0 );
    SSIM      =  cal_ssim( O_Img, E_Img, 0, 0 );
    fprintf( 'Iter = %2.3f, PSNR = %2.2f, SSIM = %2.2f \n', ite, PSNR, SSIM );
    Par.PSNR(ite, Par.image)  =   PSNR;
    Par.SSIM(ite, Par.image)      =  SSIM;
end
return;





