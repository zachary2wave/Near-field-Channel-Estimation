clear,clc
% Y  receive signal at RF chain
% Y_ the receive signal at anntena(with noise)
% Q the weithgt 
% label the sparse channel
% label_h the receive signal at anntena(without noise)
% dic  the dictionary
for snr = 1:5:30
    load(['Ongrid_snr',num2str(snr),'.mat'])
    fprintf('snr %d', snr);
    for sample =1:3840 
        label_norm = sum((real(label_h).^2) + (imag(label_h).^2), 2);
        sendin = Y(sample,:);
        nonzeros_label = sum(label(sample,:)~=0);
        for iii = 1:6
            [t] = OMP_function(sendin.', Q*dic, iii);
            eee = dic*t/1e3 - label_h(sample,:).';
            exx(iii) = sum(real(eee).^2 + imag(eee).^2)/label_norm(sample);
        end
        nonzeros = find(exx == min(exx));
        errorOMP(snr, sample) = min(exx);
    end   
end

result = sum(errorOMP,2)./3840;




