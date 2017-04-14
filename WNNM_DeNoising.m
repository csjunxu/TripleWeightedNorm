function [E_Img, Par]    =  WNNM_DeNoising( N_Img, O_Img, Par )


E_Img           = N_Img;                                                        % Estimated Image
[h, w, ch]  = size(E_Img);
Par.h = h;
Par.w = w;
Par.ch = ch;
Par = SearchNeighborIndex( Par );
            
for iter = 1 : Par.Iter        
    E_Img             	=	E_Img + Par.delta*(N_Img - E_Img);
    [CurPat, Sigma_arr]	=	Im2Patch( E_Img, N_Img, Par );                      % image to patch and estimate local noise variance            
    
    if (mod(iter-1,Par.Innerloop)==0)
        Par.nlsp = Par.nlsp-10;                                             % Lower Noise level, less NL patches
        NL_mat  =  Block_Matching(CurPat, Par);% Caculate Non-local similar patches for each 
        if(iter==1)
            Sigma_arr = Par.nSig * ones(size(Sigma_arr));                       % First Iteration use the input noise parameter
        end
    end       

     [EPat, W]  =  PatEstimation( NL_mat, Sigma_arr, CurPat, Par );   % Estimate all the patches
     E_Img      =  PGs2Image( EPat, W, Par);             
     PSNR  = csnr( O_Img, E_Img, 0, 0 );    
    fprintf( 'Iter = %2.3f, PSNR = %2.2f \n', iter, PSNR );
end
return;


