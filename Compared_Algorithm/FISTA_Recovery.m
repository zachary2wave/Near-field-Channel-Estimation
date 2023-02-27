function x = FISTA_Recovery(A,b,opt,label)
% The objective function: F(x) = 1/2 ||y - hx||^2 + lambda |x|
% Input: 
%   h: impulse response (lexicographically arranged)
%   y: degraded image (vector)
%   x0: initialization (vector)
%   opt.lambda: weight constant for the regularization term
%   hfun: functions
% Output:
%   x: output image
%
% Author: Seunghwan Yoo

display =0;
if(display)
fprintf(' - Running ISTA Method\n');
end

AtA = A'*A;
evs = eig(AtA);
L = max(evs);

lambda = opt.lambda;% 0.01
maxiter = opt.maxiter; %2000;
tol = opt.tol;%10^(-6)
vis = opt.vis;%0


xk = zeros(size(A,2),1) ;
% xk = A\b ;


% k-th (k=0) function, gradient, hessian
objk  = func(xk,b,A,lambda);
%gradk = grad(x0,b,A);


% tic;
% if(display)
% fprintf('%6s %9s %9s\n','iter','f','sparsity');
% fprintf('%6i %9.2e %9.2e\n',0,objk,nnz(xk)/numel(xk));
% end

G = A'*A ;
c = A'*b ;
t_k = 1 ; 
t_km1 = 1 ;
xkm1 = xk;
for i = 1:maxiter
    x_old = xk;
    
%     yk = xk;
    yk = xk + ((t_km1-1)/t_k)*(xk-xkm1) ;
    
    temp = G*yk - c ; % gradient of f at yk
    gk = yk - (1/L)*temp ;
    %y = xk - l*A'*(A*xk-b);
    xkp1 = subplus(abs(gk)-lambda/L) .* sign(gk); % shrinkage operation

    t_kp1 = 0.5*(1+sqrt(1+4*t_k*t_k)) ;
    
    t_km1 = t_k ;
    t_k = t_kp1 ;
    xkm1 = xk ;
    xk = xkp1 ;
    
    xk = xkp1 ;
    
    if(display)
        if vis > 0
            fprintf('%6i %9.2e %9.2e\n',i,func(xk,b,A,lambda),nnz(xk)/numel(xk));
        end

        if norm(xk-x_old)/norm(x_old) < tol
            fprintf('%6i %9.2e %9.2e\n',i,func(xk,b,A,lambda),nnz(xk)/numel(xk));
            fprintf('  converged at %dth iterations\n',i);
            break;
        end
    end
%     error(i) = sum(abs(xk - label.'));
    
end
% figure(1)
% plot(error)
% figure(2)
% hold on
% stem(xk)
% stem(label)


% toc;
x = xk;





function objk = func(xk,b,A,lambda)
e = b - A*xk;
objk = 0.5*(e)'*e + lambda*sum(abs(xk));
