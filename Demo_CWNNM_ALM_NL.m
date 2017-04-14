clear;
Original_image_dir  =    'C:\Users\csjunxu\Desktop\JunXu\Datasets\kodak24\kodak_color\';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

nSig = [40 20 30];

Par.nSig      =   nSig;                                 % Variance of the noise image
Par.win =   20;                                   % Non-local patch searching window
Par.delta     =   0;                                  % Parameter between each iter
Par.Constant         =  2 * sqrt(2);                              % Constant num for the weight vector
Par.Innerloop =   2;                                    % InnerLoop Num of between re-blockmatching
Par.ps       =   6;                            % Patch size
Par.step        =   5;
Par.Iter          =   4;                            % total iter numbers

Par.display = true;
% Par.method = 'WNNM_ALM';
Par.method = 'CWNNM_ALM';
Par.model = 'IALM';
Par.maxIter = 10;
for rho = [3]
    Par.rho = rho;
    for lambda = [4]
        Par.lambda = [4 4 4 lambda];
        % record all the results in each iteration
        Par.PSNR = zeros(Par.Iter, im_num, 'single');
        Par.SSIM = zeros(Par.Iter, im_num, 'single');
        for i = 2:im_num
            Par.image = i;
            Par.nSig0 = nSig;
            Par.nlsp        =   70;                           % Initial Non-local Patch number
            Par.I =  double( imread(fullfile(Original_image_dir, im_dir(i).name)) );
            S = regexp(im_dir(i).name, '\.', 'split');
            [h, w, ch] = size(Par.I);
            Par.nim = zeros(size(Par.I));
            for c = 1:ch
                randn('seed',0);
                Par.nim(:, :, c) = Par.I(:, :, c) + Par.nSig0(c) * randn(size(Par.I(:, :, c)));
            end
            %
            fprintf('%s :\n',im_dir(i).name);
            PSNR =   csnr( Par.nim, Par.I, 0, 0 );
            SSIM      =  cal_ssim( Par.nim, Par.I, 0, 0 );
            fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
            %
            time0 = clock;
            [im_out, Par] = CWNNM_ALM_NL_Denoising( Par.nim, Par.I, Par ); % WNNM denoisng function
            fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
            im_out(im_out>255)=255;
            im_out(im_out<0)=0;
            % calculate the PSNR
            Par.PSNR(Par.Iter, Par.image)  =   csnr( im_out, Par.I, 0, 0 );
            Par.SSIM(Par.Iter, Par.image)      =  cal_ssim( im_out, Par.I, 0, 0 );
            %             imname = sprintf('nSig%d_clsnum%d_delta%2.2f_lambda%2.2f_%s', nSig, cls_num, delta, lambda, im_dir(i).name);
            %             imwrite(im_out,imname);
            fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name, Par.PSNR(Par.Iter, Par.image),Par.SSIM(Par.Iter, Par.image)     );
        end
        mPSNR=mean(Par.PSNR,2);
        [~, idx] = max(mPSNR);
        PSNR =Par.PSNR(idx,:);
        SSIM = Par.SSIM(idx,:);
        mSSIM=mean(SSIM,2);
        fprintf('The best PSNR result is at %d iteration. \n',idx);
        fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR(idx),mSSIM);
        name = sprintf([Par.method '_NL_nSig' num2str(nSig(1)) num2str(nSig(2)) num2str(nSig(3)) '_maxiter' num2str(Par.maxIter) '_delta' num2str(Par.delta) '_lambda' num2str(lambda) '_rho' num2str(rho) '.mat']);
        save(name,'PSNR','SSIM','mPSNR','mSSIM');
    end
end