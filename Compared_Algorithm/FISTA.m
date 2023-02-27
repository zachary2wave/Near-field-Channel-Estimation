clear,clc
% Y  receive signal at RF chain
% Y_ the receive signal at anntena(with noise)
% Q the weithgt 
% label the sparse channel
% label_h the receive signal at anntena(without noise)
% dic  the dictionary

opt.lambda = 10000;
opt.maxiter = 3000;
opt.tol = 1.0000e-04;
opt.vis = 0;


for snr = 0:30
    load(['Ongrid_snr',num2str(snr),'.mat'])
    fprintf('snr %d', snr);
    for sample =1:3840 
        label_norm = sum((real(label_h).^2) + (imag(label_h).^2), 2);
        sendin = Y(sample,:).';
        nonzeros_label = sum(label(sample,:)~=0);
       for ii = 1:10
         opt.lambda =  ii/10;
         x = FISTA_Recovery(Q*dic, sendin, opt, label(sample,:));
         e = dic*x/1e3 - label_h(sample,:).';
         errorFISTA_(sample, ii) = sum(real(e).^2 + imag(e).^2)/label_norm(sample); 
       end 
        error_record(snr, sample) = min(errorFISTA_(sample, :));
    end   
end

