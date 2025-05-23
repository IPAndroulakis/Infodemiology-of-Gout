function myLinearRegressionCorrCoef(x,y, descriptors)
    % Perform linear regression and correlation analysis with visual output
    
    % Sort data for plotting and consistency
    [xs, idx] = sort(x);
    ys = y(idx);
    ds = descriptors(idx);
    x = xs;
    y = ys;
    descriptors = ds;

    n = length(x); % Number of observations
    np = 2;        % Number of model parameters
    
    %% CORRELATION ANALYSIS
    [R_pearson, P_pearson] = corr(x(:), y(:), 'Type', 'Pearson');
    [R_spearman, P_spearman] = corr(x(:), y(:), 'Type', 'Spearman');
    
    % Fisher Z transform for 95% CI (Pearson)
    z_pearson = 0.5 * log((1 + R_pearson) / (1 - R_pearson));
    SE_z = 1 / sqrt(n - 3);
    z_CI_pearson = z_pearson + [-1, 1] * 1.96 * SE_z;
    CI_pearson = tanh(z_CI_pearson);
    
    % Fisher Z transform for 95% CI (Spearman)
    z_spearman = 0.5 * log((1 + R_spearman) / (1 - R_spearman));
    z_CI_spearman = z_spearman + [-1, 1] * 1.96 * SE_z;
    CI_spearman = tanh(z_CI_spearman);
    
    %% LINEAR REGRESSION
    p = polyfit(x, y, 1);        % [slope, intercept]
    y_est = polyval(p, x);
    residuals = y - y_est;

    sse = sum(residuals.^2);     % Sum of squared errors
    df = n - np;                 % Degrees of freedom
    s2 = sse / df;               % Residual variance
    se = sqrt(s2);               % Standard error of regression

    sx2 = sum((x - mean(x)).^2); % Sum of squares of x deviations
    se_slope = sqrt(s2 / sx2);   % Standard error of slope
    t_slope = p(1) / se_slope;   % t-statistic for slope
    p_value = 2 * (1 - tcdf(abs(t_slope), df)); % two-tailed p-value

    % 95% CI for slope
    t_crit = tinv(0.975, df);
    slope_CI = [p(1) - t_crit * se_slope, p(1) + t_crit * se_slope];

    % R²
    ss_total = sum((y - mean(y)).^2);
    R_squared = 1 - (sse / ss_total);

    %% REPORT
    fprintf('\n*** Linear Regression: Peak Month vs Latitude ***\n');
    fprintf('Slope: %.4f\n', p(1));
    fprintf('95%% CI for Slope: [%.4f, %.4f]\n', slope_CI(1), slope_CI(2));
    fprintf('Intercept: %.4f\n', p(2));
    fprintf('Standard Error (regression): %.4f\n', se);
    fprintf('R-squared: %.4f\n', R_squared);
    fprintf('P-value (slope significance): %.4f\n', p_value);
    if p_value < 0.05
        fprintf('→ The slope is statistically significant (p < 0.05), indicating a meaningful linear relationship.\n');
    else
        fprintf('→ The slope is NOT statistically significant (p ≥ 0.05), suggesting weak or no linear trend.\n');
    end

    fprintf('\n--- Correlation Results ---\n');
    fprintf('Pearson r = %.4f (95%% CI: %.4f to %.4f), p = %.4f\n', R_pearson, CI_pearson(1), CI_pearson(2), P_pearson);
    if P_pearson < 0.05
        fprintf('→ Pearson correlation is statistically significant, indicating a linear association.\n');
    else
        fprintf('→ Pearson correlation is NOT statistically significant, linear association may be weak.\n');
    end
    
    fprintf('Spearman rho = %.4f (95%% CI: %.4f to %.4f), p = %.4f\n', R_spearman, CI_spearman(1), CI_spearman(2), P_spearman);
    if P_spearman < 0.05
        fprintf('→ Spearman correlation is statistically significant, indicating a monotonic relationship.\n');
    else
        fprintf('→ Spearman correlation is NOT statistically significant, monotonic trend may be weak or absent.\n');
    end

    %% PLOTTING
    figure;
    hold on;
    plot(x, y, 'go','MarkerFaceColor', 'g'); % Actual data
    plot(x, y_est, 'r-', 'LineWidth', 2);    % Fitted line

    % 95% confidence intervals around the regression line
    x_mean = mean(x);
    s_x = std(x);
    SE_pred = se * sqrt(1 + 1/n + ((x - x_mean).^2 / ((n - 1) * s_x^2)));
    intervals = t_crit * SE_pred;
    upper_bound = y_est + intervals;
    lower_bound = y_est - intervals;
    plot(x, upper_bound, 'r:');
    plot(x, lower_bound, 'r:');

    xlabel('$$Latitude$$', 'Interpreter', 'latex', 'FontSize', 20);
    ylabel('$$Amplitude$$', 'Interpreter', 'latex', 'FontSize', 20);
    xticks(20:5:80);
    set(gca, 'FontSize', 18);
    grid on;
    hold off;

    %% Annotate points with descriptors
    offset = 0.1;
    for i = 1:length(x)
        dx = offset;
        dy = offset;
        text(x(i) + dx, y(i) + dy, [' ' descriptors{i}], ...
            'VerticalAlignment', 'bottom', ...
            'HorizontalAlignment', 'right', ...
            'FontSize', 18);
    end
end
