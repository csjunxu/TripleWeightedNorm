function  [Y, SigmaArr]  =  Im2Patch( E_Img,N_Img, par )
TotalPatNum = (size(E_Img,1)-par.ps+1)*(size(E_Img,2)-par.ps+1);                  %Total Patch Number in the image
Y           =   zeros(par.ps*par.ps*par.ch, TotalPatNum, 'single');                      %Current Patches
N_Y         =   zeros(par.ps*par.ps*par.ch, TotalPatNum, 'single');                      %Patches in the original noisy image
k           =   0;

for l = 1:par.ch
    for i  = 1:par.ps
        for j  = 1:par.ps
            k     =  k+1;
            E_patch     =  E_Img(i:end-par.ps+i, j:end-par.ps+j, l);
            N_patch     =  N_Img(i:end-par.ps+i, j:end-par.ps+j, l);
            Y(k,:)      =  E_patch(:)';
            N_Y(k,:)    =  N_patch(:)';
        end
    end
end
SigmaArr = par.lambda * par.lamada*sqrt(abs(repmat(par.nSig^2,1,size(Y,2))-mean((N_Y-Y).^2)));          %Estimated Local Noise Level