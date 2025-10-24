function x = truncnorm_rnd(mu, sigma, a, b)
    % truncate
    if isinf(a) && isinf(b)
        x = mu + sigma * randn();
        return;
    end

    alpha = (a - mu) / sigma;
    beta  = (b - mu) / sigma;
    Phi_alpha = normcdf(alpha);
    Phi_beta  = normcdf(beta);

    % boundary restrain
    if Phi_beta - Phi_alpha < 1e-14
        x = max(a, mu);
        return;
    end

    u = rand() * (Phi_beta - Phi_alpha) + Phi_alpha;
    z = norminv(u);
    x = mu + sigma * z;
end
