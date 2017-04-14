clear;
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.JPG');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\our_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.JPG');
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

Par.win =   20;   % Non-local patch searching window
Par.delta     =   0;  % Parameter between each iter
Par.Constant         =  2 * sqrt(2);   % Constant num for the weight vector
Par.Innerloop =   2;   % InnerLoop Num of between re-blockmatching
Par.ps       =   6;   % Patch size
Par.step        =   5;
Par.Iter          =   2; % total iter numbers

Par.display = true;
% Par.method = 'WNNM_ADMM';
Par.method = 'CWNNM_ADMM';
Par.model = '2';
Par.maxIter = 10;


for mu = [1.001]
    Par.mu = mu;
    for rho = 0.002:0.002:0.01
        Par.rho = rho;
        for lambda = [4:2:8]
            Par.lambda = [lambda lambda];
            % record all the results in each iteration
            Par.PSNR = zeros(Par.Iter, im_num, 'single');
            Par.SSIM = zeros(Par.Iter, im_num, 'single');
            for i = 1:im_num
                Par.image = i;
                Par.nlsp  =  70;                           % Initial Non-local Patch number
                Par.I = double( imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)) );
                S = regexp(GT_im_dir(i).name, '\.', 'split');
                fprintf('%s :\n', GT_im_dir(i).name);
                [h, w, ch] = size(Par.I);
                Par.nim = double( imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name)) );
                for c = 1:ch
                    % Par.nSig0(c) = NoiseLevel(Par.nim(:, :, c));
                    Par.nSig0(c,1) = NoiseEstimation(Par.nim(:, :, c), Par.ps);
                end
                fprintf('The noise levels are %2.2f, %2.2f, %2.2f. \n', Par.nSig0(1), Par.nSig0(2), Par.nSig0(3) );
                PSNR =   csnr( Par.nim, Par.I, 0, 0 );
                SSIM      =  cal_ssim( Par.nim, Par.I, 0, 0 );
                fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
                time0 = clock;
                [im_out, Par] = CWNNM_ADMM_NL2_Denoising( Par.nim, Par.I, Par ); % WNNM denoisng function
                fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
                im_out(im_out>255)=255;
                im_out(im_out<0)=0;
                %    calculate the PSNR
                Par.PSNR(Par.Iter, Par.image)  =   csnr( im_out, Par.I, 0, 0 );
                Par.SSIM(Par.Iter, Par.image)      =  cal_ssim( im_out, Par.I, 0, 0 );
                imname = sprintf(['C:/Users/csjunxu/Desktop/ICCV2017/cc_Results/' Par.method '_NL_CC15_' Par.model '_Oite' num2str(Par.Iter) '_Iite' num2str(Par.maxIter) '_rho' num2str(rho) '_mu' num2str(Par.mu) '_lambda' num2str(lambda) '_' TT_im_dir(i).name]);
                imwrite(im_out/255, imname);
                fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',TT_im_dir(i).name, Par.PSNR(Par.Iter, Par.image),Par.SSIM(Par.Iter, Par.image)     );
            end
            mPSNR=mean(Par.PSNR,2);
            [~, idx] = max(mPSNR);
            PSNR =Par.PSNR(idx,:);
            SSIM = Par.SSIM(idx,:);
            mSSIM=mean(SSIM,2);
            fprintf('The best PSNR result is at %d iteration. \n',idx);
            fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR(idx),mSSIM);
            name = sprintf([Par.method '_NL2_CC' num2str(im_num) '_' Par.model '_Oite' num2str(Par.Iter) '_Iite' num2str(Par.maxIter) '_rho' num2str(rho) '_mu' num2str(Par.mu) '_lambda' num2str(lambda) '.mat']);
            save(name,'PSNR','SSIM','mPSNR','mSSIM');
        end
    end
end

