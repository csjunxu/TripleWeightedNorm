clear;
% Original_image_dir  =    'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\';
% nSig = [40 20 30];
% nSig = [5 30 15];
% nSig = [30 10 50];

Original_image_dir  =    'C:\Users\csjunxu\Desktop\Projects\WODL\20images\';
% Original_image_dir  =    'C:\Users\csjunxu\Desktop\Projects\WODL\20newimages\';


Sdir = regexp(Original_image_dir, '\', 'split');
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);


nSig = [40];

Par.nSig      =   nSig;                                 % Variance of the noise image
Par.win =   20;                                   % Non-local patch searching window
Par.Constant         =  2 * sqrt(2);                              % Constant num for the weight vector
Par.ps       =   7;                            % Patch size
Par.step        =   6;
Par.Outerloop  =   8;                            % total iter numbers
Par.Innerloop =   2;                                    % InnerLoop Num of between re-blockmatching

Par.display = true;


% Par.method = 'WNNM_ADMM';
Par.method = 'W3NNM_ADMM';
Par.maxIter = 10;

Par.delta     =   0;                                  % Parameter between each iter
for mu = [1.001 1.01]
    Par.mu = mu;
    for rho = [0.05 0.1]
        Par.rho = rho;
        for lambda1 = 0.3:0.1:1
            Par.lambda1 = lambda1;
            for lambda2 = 0.3:0.1:1
                Par.lambda2 = lambda2;
                % record all the results in each iteration
                Par.PSNR = zeros(Par.Outerloop, im_num, 'single');
                Par.SSIM = zeros(Par.Outerloop, im_num, 'single');
                for i = 1:im_num
                    Par.image = i;
                    Par.nSig = nSig;
                    Par.nlsp        =   90;   % Initial Non-local Patch number
                    Par.I =  double( imread(fullfile(Original_image_dir, im_dir(i).name)) );
                    S = regexp(im_dir(i).name, '\.', 'split');
                    [h, w, ch] = size(Par.I);
                    Par.nim = zeros(size(Par.I));
                    for c = 1:ch
                        randn('seed',0);
                        Par.nim(:, :, c) = Par.I(:, :, c) + Par.nSig(c) * randn(size(Par.I(:, :, c)));
                    end
                    %                 imname = sprintf(['C:/Users/csjunxu/Desktop/ICCV2017/24images/Noisy_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_' im_dir(i).name]);
                    %                 imwrite(Par.nim/255, imname);
                    fprintf('%s :\n',im_dir(i).name);
                    PSNR =   csnr( Par.nim, Par.I, 0, 0  );
                    SSIM      =  cal_ssim( Par.nim, Par.I, 0, 0 );
                    fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
                    %
                    time0 = clock;
                    [im_out, Par] = W3NNM_ADMM_Denoising( Par.nim, Par.I, Par );
                    fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
                    im_out(im_out>255)=255;
                    im_out(im_out<0)=0;
                    % calculate the PSNR
                    Par.PSNR(Par.Outerloop, Par.image)  =   csnr( im_out, Par.I, 0, 0 );
                    Par.SSIM(Par.Outerloop, Par.image)      =  cal_ssim( im_out, Par.I, 0, 0 );
                    %                 imname = sprintf(['C:/Users/csjunxu/Desktop/ICCV2017/24images/' Par.method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_Oite' num2str(Par.Outerloop) '_Iite' num2str(Par.maxIter) '_rho' num2str(rho) '_mu' num2str(mu) '_lambda' num2str(lambda) '_' im_dir(i).name]);
                    %                 imwrite(im_out/255, imname);
                    fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name, Par.PSNR(Par.Outerloop, Par.image),Par.SSIM(Par.Outerloop, Par.image)     );
                end
                mPSNR=mean(Par.PSNR,2);
                [~, idx] = max(mPSNR);
                PSNR =Par.PSNR(idx,:);
                SSIM = Par.SSIM(idx,:);
                mSSIM=mean(SSIM,2);
                fprintf('The best PSNR result is at %d iteration. \n',idx);
                fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR(idx),mSSIM);
                %             name = sprintf([Par.method '_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_Oite' num2str(Par.Outerloop) '_Iite' num2str(Par.maxIter) '_rho' num2str(rho) '_mu' num2str(mu) '_lambda' num2str(lambda) '.mat']);
                name = sprintf([Par.method '_nSig' num2str(nSig) '_Oite' num2str(Par.Outerloop) '_Iite' num2str(Par.maxIter) '_rho' num2str(rho) '_mu' num2str(mu) '_lambda1' num2str(lambda1) '_lambda2' num2str(lambda2) '.mat']);
                save(name,'nSig','PSNR','SSIM','mPSNR','mSSIM');
            end
        end
    end
end