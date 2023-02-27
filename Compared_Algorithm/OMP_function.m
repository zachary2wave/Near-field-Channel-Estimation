function [t] = OMP_function(x, D, L)
% x: input signal
% D: dictionary, each column is an atom. Note: atoms much be normalized.
% threshold: stop conditions
% L: number of non-zero entries in output
% t:
%   output, a vector that stores the coefficient of each atom

% Demo: 
% dictionary = [1 0; 1/2 sqrt(3)/2; -1/sqrt(2) -1/sqrt(2)]';
% x = [1 1/2]';
% t = omp(x, dictionary, 2);
% output_x = D * t;


dim = size(D,2);
t = zeros(dim,1);
r = x; % residual
selected_atom = [];
for iter = 1 : L
    dot_p = D' * r;
    [~, i] = max(abs(dot_p));
    selected_atom = [selected_atom i];
    temp = D(:, selected_atom);
    a = pinv(temp) * x;
    r = x - temp * a;
%     fprintf('Iter %d, residual: %f\n', iter, norm(r,2));
end
t(selected_atom) = a;