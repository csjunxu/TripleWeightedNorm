clear;
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\1_Results\Real_NoisyImage\';
TT_fpath = fullfile(TT_Original_image_dir, '*.png');
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

Par.win =   20;   % Non-local patch searching window
Par.delta     =   0.1;  % Parameter between each iter
Par.Constant         =  2 * sqrt(2);   % Constant num for the weight vector
Par.Innerloop =   2;   % InnerLoop Num of between re-blockmatching
Par.ps       =   6;   % Patch size
Par.step        =   5;
Par.Iter          =   2; % total iter numbers

Par.display = true;
Par.method = 'WNNM_ADMM';
% Par.method = 'CWNNM_ADMM';
Par.model = 'ADMM';
Par.maxIter = 10;
for lambda = 1.2:0.1:2
    Par.lambda = [lambda lambda];
    for rho = [6]
        Par.rho = rho;
        for mu = [1.001]
            Par.mu = mu;
            % record all the results in each iteration
            Par.PSNR = zeros(Par.Iter, im_num, 'single');
            Par.SSIM = zeros(Par.Iter, im_num, 'single');
            for i = im_num:-1:1
                Par.image = i;
                Par.I = double( imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name)) );
                Par.nim = double( imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name)) );
                Par.nlsp  =  70;                           % Initial Non-local Patch number
                S = regexp(TT_im_dir(i).name, '\.', 'split');
                fprintf('%s :\n', TT_im_dir(i).name);
                [h, w, ch] = size(Par.I);
                for c = 1:ch
                    % Par.nSig0(c) = NoiseLevel(Par.nim(:, :, c));
                    Par.nSig0(c,1) = NoiseEstimation(Par.nim(:, :, c), Par.ps);
                end
                fprintf('The noise levels are %2.2f, %2.2f, %2.2f. \n', Par.nSig0(1), Par.nSig0(2), Par.nSig0(3) );
                time0 = clock;
                [im_out, Par] = CWNNM_ALM_NL_Denoising( Par.nim, Par.I, Par ); % WNNM denoisng function
                fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
                im_out(im_out>255)=255;
                im_out(im_out<0)=0;
                imname = sprintf(['C:/Users/csjunxu/Desktop/ICCV2017/1nc_Results/' Par.method '_NL_NC_' Par.model '_Oite' num2str(Par.Iter) '_Iite' num2str(Par.maxIter) '_rho' num2str(rho) '_mu' num2str(Par.mu) '_lambda' num2str(lambda) '_' TT_im_dir(i).name]);
                imwrite(im_out/255, imname);
            end
        end
    end
end