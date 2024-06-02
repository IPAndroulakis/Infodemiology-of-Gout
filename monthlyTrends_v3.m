close all
clear all
global Pval
Pval = 0.05;
% add path of common MATLAB function
addpath('C:\Users\ipand\Documents\MY.DOCUMENTS\Research\MY.MATLAB');
addpath('C:\Users\ipand\Documents\MY.DOCUMENTS\Research\MY.COLLABORATIONS\Gout\Google trends\shapefile');
% fres    = fopen('C:\Users\ipand\Documents\MY.DOCUMENTS\Research\MY.COLLABORATIONS\Gout\Google trends\myositisCSV\ResultsFigures\myositis.txt', 'w');
% resultsDir = 'C:\Users\ipand\Documents\MY.DOCUMENTS\Research\MY.COLLABORATIONS\Gout\Google trends\myositisCSV\ResultsFigures\';
%folderPath = 'myositisCSV'; % Change this to your folder path
%%%%%%
%%% Define your keyword: 
%keyword = 'muscleWeak'; 
%nR = 1; nC = 5;
keyword = input('Enter keyword: ', 's');

%%%%%%
% Base Directory (modify as needed)
baseDirectory = 'C:\Users\ipand\Documents\MY.DOCUMENTS\Research\MY.COLLABORATIONS\Gout\Google trends\';
% Create the folderPath
folderPath = [keyword, 'CSV'];  
% Construct directory and file names
resultsDir = fullfile(baseDirectory, folderPath, 'ResultsFigures');
fileName = fullfile(resultsDir, [keyword, '.txt']);
% Create directory if it doesn't exist
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end
% Open the file for writing
fres = fopen(fileName, 'w');
%%%%%%
fprintf(fres, '%3s %35s %6s %7s %7s %6s %6s %6s %4s %8s\n', '###', 'Full Name', 'ISO', 'LAT', 'LONG', 'MESOR', 'AMPL', 'PHASE', 'MNTH', 'Pval'); 
%% process state data
% Specify the directory containing the Excel files

% Get a list of all files in the folder with the .xlsx extension
files = dir(fullfile(folderPath, '*.csv')); %was .xlsx
n = length(files);
fprintf('number of .csv files found: %d\n', n)
nR = input('Enter number or rows per plot: ');
nC = input('Enter number or columns per plot: ');
% allocate subplots subplot
fig = figure();
ic = 1;
for i=1:nR
    for j=1:nC
        l = (i-1)*nC+j;
        ax(l) = subplot(nR, nC, l);
    end
end
ik = 0;
for k = 1:length(files)
    stateName = files(k).name;
    stateName = stateName(1:end-4); % Remove the last 5 characters '.xlsx' or 4 for .csv
    countriesToColor{k} = stateName;
    fullPath = fullfile(files(k).folder, files(k).name);   
    filename = fullfile(files(k).folder, files(k).name);
    % the lines bellow will extract the actual name of the city, state,
    % country
        fileContent = fileread(filename);
        % Use regular expressions to find the country name within parentheses
        countryPattern = '\((.*?)\)'; % This pattern captures any characters inside parentheses
        tokens = regexp(fileContent, countryPattern, 'tokens');
        fullName = tokens{1}{1}; % This will give you 'United Arab Emirates'
    % the line below will skip the first 3 lines
        opts = detectImportOptions(filename, 'HeaderLines', 3); % skip the first three lines
        data = readtable(filename, opts);
    fprintf('\n*** full name %s -- ISO abbreviation %s\n', fullName, stateName)
    % process the data
    [h(k) M(k) A(k) phi(k) pValue(k) month(k)] = googleTrends(data, ic, stateName, fullName, fig, ax(ic), nR, nC);  
    if mod(k,nR*nC)==0 && k<length(files)
        ic = 1;
        fig = figure();
        % re-allocate subplots subplot
        ic = 1;
        for i=1:nR
            for j=1:nC
                l = (i-1)*nC+j;
                ax(l) = subplot(nR, nC, l);
            end
        end
    else
        ic = ic+1;
    end
    acrophase_months = mod(mod(phi(k) - 2, 12), 12) + 1;
    %% get lattitude and longitude - change whether city, state of country
    if strcmp(keyword, 'World10')
        [lat(k), longi(k)] = getCountryLatitude(stateName);
    elseif strcmp(keyword,'USstates')  
        [lat(k), longi(k)] = getStateLatitude(stateName);
    else
        [lat(k), longi(k)] = getCityLatitude(stateName);
    end
    fprintf('Peaking month %3.0f\n', month(k));
    if pValue(k) < Pval
        fprintf(fres, '%3d %35s %6s %7.2f %7.2f %6.2f %6.2f %6.2f %4.1f %4.2e (***)\n', k, fullName, stateName, lat(k), longi(k), M(k), A(k), phi(k), month(k), pValue(k));
    else
        fprintf(fres, '%3d %35s %6s %7.2f %7.2f %6.2f %6.2f %6.2f %4.1f %4.2e\n',       k, fullName, stateName, lat(k), longi(k), M(k), A(k), phi(k), month(k), pValue(k));
    end 
    % Spring = 1; Summer = 2; Fall = 3; Winter = 4;
    if month(k)>3 && month(k) <= 5
        season(k) = 1;
    elseif month(k)>5 && month(k) <= 8
        season(k) = 2;
    elseif month(k)>8 && month(k) <= 11
        season(k) = 3;
    else
        season(k) = 4;
    end

    Sname{k} = stateName;
    if pValue(k) < Pval  
        ik = ik+1;
        aA(ik) = A(k);
        rlat(ik) = lat(k);
        rlongi(ik) = longi(k);
        rmonth(ik) = month(k);
        Fname{ik} = fullName;
    end
end
%% linear regression
if strcmp(keyword, 'UScities')
    myLinearRegression(rlat, aA, Fname);
end
%% get surface plot (lat, log, month)
%mysufPlot(rlat, rlongi, rmonth);
%% multilinear regression 
%myMultiLinearRegression(lat, longi, month);
%% Get world map
getWorldMap(countriesToColor, pValue, season)
%% save all files to USstates/figs   
figHandles = findall(groot, 'Type', 'figure'); 
for i = 1:length(figHandles)
    filename = get(figHandles(i), 'Name');
    if isempty(filename)
        filename = sprintf('figure_%d', i); % Assign names for untitled figures
    end
    
    % Save as .fig
    fullSavePathFig = fullfile(resultsDir, [filename, '.fig']);
    savefig(figHandles(i), fullSavePathFig);

    % Save as .jpg, making sure the size is enlarged and readable
    figure(figHandles(i)); % Make sure the current figure is active
    set(figHandles(i), 'Units', 'Inches', 'Position', [0, 0, 8, 6]); % Resize the figure (8x6 inches for example)
    fullSavePathJPG = fullfile(resultsDir, [filename, '.jpg']);
    saveas(figHandles(i), fullSavePathJPG, 'jpeg');
end
createPDF(folderPath)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% create polar plot
% creat_polar_plot(t_max, Sname)
% plot(abs(A), phi, 'or')
%legend(h, {'MN (46oN)', 'FL (28oN)', 'NJ (38oN)', 'CA (40oN)'})
%% Process trend data
function [h M A phi p_value t_max] = googleTrends(data, ic, stateName, fullName, fig, ax, nR, nC)
global Pval
fig;
axes(ax);
%% the lines bellow are for reading all years
% % Preallocate arrays based on the number of rows in the data
% years = zeros(height(data), 1);
% months = zeros(height(data), 1);
% values = zeros(height(data), 1);
% % Loop through each row to populate the arrays
% for i = 1:height(data)
%     dateStr = data.Var1{i}; % Replace 'Var1' with the actual variable name if different
%     values(i) = data.Var2(i); % Replace 'Var2' with the actual variable name if different    
%     % Split the date string and extract year and month
%     dateParts = strsplit(dateStr, '-');
%     years(i) = str2double(dateParts{1});
%     months(i) = str2double(dateParts{2});
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% the lines bellow are checking to make sure we only process years > 2014
% Initialize arrays to store the values, months, and years.
% Assuming that 'data' has already been read and contains the necessary information.
% You should also pre-allocate these arrays if you know the number of entries.
% For now, let's assume we don't know the number of entries to expect.
values = [];
years = [];
months = [];
for i = 1:height(data)
    dateStr = data.Var1{i}; % Replace 'Var1' with the actual variable name if different
    dateParts = strsplit(dateStr, '-');
    year = str2double(dateParts{1});
    month = str2double(dateParts{2});    
    % Check if the year is 2014 or later
    if year >= 2014
        values(end+1) = data.Var2(i); % Replace 'Var2' with the actual variable name if different    
        years(end+1) = year;
        months(end+1) = month;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(years);
sums = zeros(1,12);
valspermonth = zeros(1,12);
for i=1:12
    SpM(i).n = 0;
    SpM(i).S = [];
end
for i=1:n
   sums(months(i)) = sums(months(i)) + values(i);
   valspermonth(months(i)) = valspermonth(months(i))+1;
   SpM(months(i)).n = SpM(months(i)).n+1;
   SpM(months(i)).S = [SpM(months(i)).S values(i)];
end
% check to make sure that all months have all years
mY = -1;
for i=1:12
    if SpM(i).n > mY
        mY = SpM(i).n;
    end
end
% add one more entry to months missing a year (we are now in April, so
% 2024 has one more measurements for the 4 months than the rest of the
% years)
for i=1:12
    if SpM(i).n < mY
        SpM(i).S = [SpM(i).S round(mean(SpM(i).S))];
    end
end

for i=1:12
    sums(i) = sums(i)/valspermonth(i);
end
%% get box plot of data and calculate means
for i=1:12   
    data = SpM(i).S;
    data_imputed = data;
    dataA(:,i) = data_imputed;
    in = -1; % if in = 1, then do data imputation 
    if in == 1     
        non_zero_mean = mean(data(data~=0));
        std_dev = std(data(data~= 0)); % Estimate standard deviation 
        random_noise = std_dev * 0.2 * randn(size(find(data==0))); % Adjust 0.2 for noise level
        data_imputed(data_imputed == 0) = non_zero_mean + random_noise; 
        dataA(:,i) = data_imputed;   
    end
    % outlier removal for box plot
    data_without_outliers = data_imputed;
    im = -1; % if im = 1 or 2, do the outlier detection
    if im == 1
        Q1 = quantile(data_imputed, 0.25);
        Q3 = quantile(data_imputed, 0.75);
        IQR = Q3 - Q1;
        outlier_threshold = 1.5 * IQR; 
        lower_bound = Q1 - outlier_threshold;
        upper_bound = Q3 + outlier_threshold;
        outliers = data_inputed < lower_bound | data_inputed > upper_bound; 
        data_without_outliers = data_imputed(~outliers);
    elseif im == 2
        outliers = isoutlier(data_imputed, 'quartiles'); 
        data_without_outliers = data_imputed(~outliers);
        data_without_outliers = data_without_outliers(data_without_outliers~=0);    
    end
    boxplot(data_without_outliers, 'positions', i); hold on; 
    means(i) = mean(data_without_outliers);
    plot(means, 'or', 'LineWidth', 0.5);
end

%% cosinor analysis
    % dataA = each column is a month
    % for i=1:12
    %     dataA(:,i) = SpM(i).S;
    % end
    t = 1:12;  % Define the time vector for one period over 12 points
    %x = means;
    % option (1) the equation bellow is the standard cosint analysis
    % cosinor_fun = @(b, t) b(1) + b(2) * sin(2 * pi * t / 12 + b(3)); %[IPA] changed to sin
    % option (2) the equation bellow was suggested as a way to make sure that 
    % 1 <= phase <= 12
    % cosinor_fun = @(b, t) b(1) + b(2) * sin(2 * pi * t / 12 + mod(b(3), 12)); 
    means = mean(dataA);
    x = means;   
    stds = std(dataA);
    bestSSE = 1.e10;
    %for i=1:10
    phiT = -1;
    amt = -1;
    while phiT<0 || amt<0
        initial_guess(1) = mean(data) + 0.5 * std(data) * randn();  % Mesor
        initial_guess(2) = 0.5 * range(data) * rand();              % Amplitude
        initial_guess(3) = 2 * pi * rand();                         % Phase
        cosinor_fun = @(b, t) b(1) + b(2) * sin(2 * pi * t / 12 + b(3));
        [beta, R, J, CovB, mse] = nlinfit(t, means, cosinor_fun, initial_guess, 'Weights', 1./stds.^2); 
        % Adjust the phase beta(3) to be within 0 to 2*pi range
        beta(3) = mod(beta(3), 2 * pi);        
        phiT = beta(3);
        amt = beta(2);
        SSE = mse * (length(means) - length(beta));
        %if SSE < bestSSE
            bestSSE = SSE;
            bestFitParameters = beta;
            bestCovB = CovB;
        %end
    end
    beta = bestFitParameters;
    CovB = bestCovB;
    % Locate the max
        % Fine-grained time points within the range
        t_vals = linspace(1, 12, 1000);  % Use a sufficient number of points    
        % Evaluate the function
        vals = cosinor_fun(beta, t_vals);  
        % Find the index of the maximum and corresponding 't' value
        [~, idx] = max(vals);
        t_max = t_vals(idx); 
    % Calculate statistics
    SE = sqrt(diag(CovB));  % Standard errors of the parameter estimates
    t_stat = beta(2) / SE(2);  % t-statistic for amplitude
    p_value = 2 * (1 - tcdf(abs(t_stat), length(x) - length(beta)));  % Two-tailed p-value
    % Display results
    fprintf('* Cosinor analysis\n');
    fprintf('Mesor (M): %f\n', beta(1));
    fprintf('Amplitude (A): %f\n', beta(2));
    fprintf('Acrophase (Phi): %f radians\n', beta(3));
    fprintf('t-statistic for Amplitude: %f\n', t_stat);
    fprintf('p-value for Amplitude: %f\n', p_value);
    A = beta(2); phi = beta(3); M = beta(1);
    xFine = linspace(1, 12, 100);
    yFine = cosinor_fun(beta, xFine);
    % Plot the fitted sinusoid
    h = plot(xFine, yFine, '-r');
%% fix the plot
if p_value < Pval
    tt = sprintf('%3s***', fullName);
else
    tt = sprintf('%3s', fullName);
end
ylabel(tt)

currentPlotIndex = ic;   % Index of the current subplot (1, 2, 3, ...)
rowPosition = floor((currentPlotIndex - 1) / nC) + 1; % Calculates row position
columnPosition = mod(currentPlotIndex - 1, nC) + 1;  % Calculates column number
if rowPosition == nR 
    xticks(1:12);
    xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});
else
    xticks(1:12);
    set(gca, 'XTickLabel', []); 
end  
ylim([0 100]);
yticks(0:10:100);
% if columnPosition == 1 
%     ylim([0 110])
%     yticks(0:10:110);
% else
%     ylim([0 110])
%     yticks(0:10:110);
%     set(gca, 'YTickLabel', []); 
% end 
grid on;
end
