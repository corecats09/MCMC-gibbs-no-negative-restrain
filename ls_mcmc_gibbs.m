% input component matrix X ~  n x p ; cell spectrum Y ~ n x 1;
% upload truncnorm_rnd.m before run this script
% MCMC via Gibbs sampling with non-negative restraint

[n, p] = size(X);
assert(size(Y,1)==n && size(Y,2)==1, 'Y must be n x 1');

% non information prior
mu0 = zeros(p,1);
V0  = eye(p)*1e6;   
a0  = 1e-3;
b0  = 1e-3;

% MCMC setup 
nIter = 12000;
burnin = 2000;
beta_samples = zeros(p, nIter - burnin);
sigma2_samples = zeros(1, nIter - burnin);

% initialisation
beta = max(0, randn(p,1));
sigma2 = 1;

% preassignments
XtX = X' * X;
XtY = X' * Y;
V0inv = inv(V0);

% timing
tic; 

% MCMC Gibbs loop
save_idx = 0;
for t = 1:nIter

    
    Vn = inv(XtX + V0inv);              % p x p
    mun = Vn * (XtY + V0inv * mu0);     % p x 1
    Sigma = sigma2 * Vn;                % Σ = σ^2 Vn

   
    for j = 1:p
        idx = [1:j-1, j+1:p];

        Sigma_jj = Sigma(j,j);
        Sigma_jm = Sigma(j, idx);       % 1 x (p-1)
        Sigma_mm = Sigma(idx, idx);     % (p-1) x (p-1)
        Sigma_mj = Sigma(idx, j);       % (p-1) x 1

       
        if isempty(idx)
            cond_mean = mun(j);
            cond_var  = Sigma_jj;
        else
            % solve Sigma_mm * x = (beta(idx)-mun(idx)) for x
            x = Sigma_mm \ (beta(idx) - mun(idx));
            cond_mean = mun(j) + Sigma_jm * x;
            % cond variance
            cond_var = Sigma_jj - Sigma_jm * (Sigma_mm \ Sigma_mj);
            % constraints
            cond_var = max(cond_var, 1e-12);
        end

        % constraints
        a = 0; b = Inf;

        % truncature
        beta(j) = truncnorm_rnd(cond_mean, sqrt(cond_var), a, b);
    end
    
    toc; 

    % renewal sigma^2 | beta, Y （Inv-Gamma） 
    an = a0 + n/2;
    resid = Y - X * beta;
    bn = b0 + 0.5 * (resid' * resid);
    sigma2 = 1 / gamrnd(an, 1/bn);

    % record
    if t > burnin
        save_idx = save_idx + 1;
        beta_samples(:, save_idx) = beta;
        sigma2_samples(save_idx) = sigma2;
    end

    % print progression
    if mod(t,2000)==0
        fprintf('Iter %d / %d\n', t, nIter);
    end
end

% posterior print
beta_mean = mean(beta_samples, 2);
beta_CI = prctile(beta_samples', [2.5 97.5]);

fprintf('\nPosterior mean of beta (non-negative constraint):\n');
disp(beta_mean);
fprintf('95%% credible intervals (rows: params, cols: lower/upper):\n');
disp(beta_CI);

