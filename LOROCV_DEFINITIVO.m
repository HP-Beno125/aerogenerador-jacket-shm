%% =========================================================
% VALIDACION CRUZADA DEJANDO UNA REPETICION FUERA (LOROCV)
% EXPERIMENTO DEFINITIVO
%
% Se evaluan los tres modelos seleccionados en el cribado preliminar:
% LDA, SVM cuadratica y ensamble bagged, sobre un barrido de cinco
% longitudes de ventana (blockLen = 50, 100, 125, 196, 245).
%
% El procedimiento incorpora los siguientes elementos:
%   (1) se registran exactitud y F1 de entrenamiento por fold, lo que
%       permite cuantificar la brecha entre entrenamiento y prueba
%   (2) se mantiene la validacion cruzada agrupada por repeticion fisica
%       (k = 3, declarado de forma explicita)
%   (3) se generan matriz de confusion, curvas ROC y curvas
%       precision-sensibilidad para los tres modelos
%   (4) los hiperparametros se declaran de forma explicita y se serializan
%   (5) la semilla del generador aleatorio se fija para garantizar
%       reproducibilidad exacta
%   (6) se incluye una linea base trivial (clasificador mayoritario)
%
% Se serializa la informacion completa del entrenamiento, esto es, los
% modelos y las transformaciones de cada fold, de manera que sea posible
% volver a predecir sin necesidad de reentrenar.
%% =========================================================

clear; clc; close all;

% ================================================================
% CONFIGURACION GLOBAL
% ================================================================
dataFolder   = 'C:\Users\52953\Desktop\experimetosfinal\REsultados para entrega de tesis\Boques_preliminar_sin_multiplicador';
inputPattern = 'bloques_con_metadata_%d.mat';   % un archivo por blockLen

blockLenList = [50, 100, 125, 196, 245];        % ventanas del barrido definitivo

SEED        = 42;        % semilla global para reproducibilidad exacta
SAVE_MODELS = true;      % si es true, se serializan los modelos y las
                         % transformaciones de cada fold, lo que permite
                         % volver a predecir sin reentrenar

% --- Modelos reportados (los tres seleccionados en el cribado preliminar) ---
modelNames  = {'LDA', 'SVM_cuadratica', 'Ensamble_bagged'};
modelLabels = {'LDA', 'SVM cuadratica', 'Ensamble bagged'};
nModels     = numel(modelNames);

% --- Hiperparametros explicitos. Se declaran aqui y se guardan como
%     metadato reproducible. La validacion cruzada anidada completa no es
%     viable con k = 3, ya que solo restarian dos repeticiones internas,
%     por lo que se fijan valores documentados en lugar de anidar. ---
hp = struct();
hp.LDA.DiscrimType         = 'linear';
hp.LDA.Gamma               = 0;          % sin regularizacion adicional, dado que
                                         % las caracteristicas del PCA ya estan
                                         % descorrelacionadas
hp.SVM_cuadratica.Kernel      = 'polynomial';
hp.SVM_cuadratica.Order       = 2;
hp.SVM_cuadratica.BoxC        = 1;       % margen estandar
hp.SVM_cuadratica.Standardize = false;   % los datos ya fueron estandarizados y
                                         % proyectados por PCA aguas arriba
hp.Ensamble_bagged.Method             = 'Bag';
hp.Ensamble_bagged.NumLearningCycles  = 50;
hp.Ensamble_bagged.MinLeafSize        = 1;

% --- Umbral de varianza retenida por el PCA ---
PCA_VAR = 95;

% ================================================================
% BUCLE EXTERNO SOBRE LAS LONGITUDES DE VENTANA
% ================================================================
for bl = blockLenList

    rng(SEED, 'twister');   % se reinicia el generador en cada ventana para
                            % asegurar determinismo

    inputFile = fullfile(dataFolder, sprintf(inputPattern, bl));
    if ~isfile(inputFile)
        warning('No se encontro %s. Se omite blockLen = %d.', inputFile, bl);
        continue;
    end

    load(inputFile, 'Xall', 'ClassID', 'RPM', 'RepID', 'blockLen', 'step');

    fprintf('\n==================================================\n');
    fprintf('LOROCV DEFINITIVO - 3 modelos | blockLen = %d\n', blockLen);
    fprintf('==================================================\n');
    fprintf('Conjunto de datos: %d bloques | step = %d\n', size(Xall,1), step);

    % ------------------------------------------------------------
    % IDENTIFICACION DE LAS REPETICIONES FISICAS
    % Constituyen la unidad de agrupamiento del esquema de validacion.
    % ------------------------------------------------------------
    PhysRepID  = strcat("C", string(ClassID), "_R", string(RepID));
    [G, ~]     = findgroups(PhysRepID);
    physClass  = splitapply(@(x) x(1), ClassID, G);
    physRepNum = splitapply(@(x) x(1), RepID,   G);

    fprintf('Repeticiones fisicas por clase:\n');
    for c = unique(physClass)'
        fprintf('  Clase %d: %d repeticiones\n', c, sum(physClass == c));
    end

    uniqueRepNumbers = unique(physRepNum);
    nFolds = numel(uniqueRepNumbers);
    fprintf('Numero de folds (repeticiones): %d\n', nFolds);
    fprintf('NOTA: el tamano muestral efectivo independiente es %d.\n\n', nFolds);

    % ------------------------------------------------------------
    % ESTRUCTURAS DE ALMACENAMIENTO
    % ------------------------------------------------------------
    accTest   = nan(nFolds, nModels);   % exactitud en prueba
    f1Test    = nan(nFolds, nModels);   % F1-macro en prueba
    accTrain  = nan(nFolds, nModels);   % exactitud en entrenamiento
    f1Train   = nan(nFolds, nModels);   % F1-macro en entrenamiento
    trainTime = nan(nFolds, nModels);
    predTime  = nan(nFolds, nModels);
    nTestObs  = zeros(nFolds, 1);
    baselineAcc = nan(nFolds, 1);       % linea base del clasificador mayoritario
    k95Fold     = zeros(nFolds, 1);
    foldResults = cell(nFolds, 1);

    % Predicciones fuera de fold, concatenadas para las figuras por modelo
    modelPredictions = struct();
    for m = 1:nModels
        modelPredictions.(modelNames{m}) = struct( ...
            'y_true', categorical([]), ...
            'y_pred', categorical([]), ...
            'scores', []);
    end

    % Contenedor de modelos y transformaciones, para evitar el reentrenamiento
    if SAVE_MODELS
        savedModels     = cell(nFolds, nModels);
        savedTransforms = cell(nFolds, 1);
    end

    % ------------------------------------------------------------
    % BUCLE DE FOLDS
    % ------------------------------------------------------------
    for fold = 1:nFolds

        testRepNum = uniqueRepNumbers(fold);

        fprintf('----------------------------------------\n');
        fprintf('FOLD %d/%d: repeticion %d reservada para prueba\n', ...
            fold, nFolds, testRepNum);
        fprintf('----------------------------------------\n');

        testMask  = (RepID == testRepNum);
        trainMask = ~testMask;

        X_train = Xall(trainMask, :);
        X_test  = Xall(testMask,  :);
        Y_train = categorical(ClassID(trainMask));
        Y_test  = categorical(ClassID(testMask));

        fprintf('Bloques de entrenamiento: %d | bloques de prueba: %d\n', ...
            size(X_train,1), size(X_test,1));

        % --- Estandarizacion z-score, ajustada unicamente con entrenamiento ---
        mu    = mean(X_train, 1);
        sigma = std(X_train, 0, 1);
        sigma(sigma == 0) = 1;
        X_train_n = (X_train - mu) ./ sigma;
        X_test_n  = (X_test  - mu) ./ sigma;

        % --- PCA, ajustado unicamente con entrenamiento ---
        warnState = warning('off', 'stats:pca:ColRankDefX');
        [coeff, scoreTrain, ~, ~, explained, muPCA] = pca(X_train_n);
        warning(warnState);
        cumExplained = cumsum(explained);
        k95 = find(cumExplained >= PCA_VAR, 1, 'first');
        X_train_pca = scoreTrain(:, 1:k95);
        X_test_pca  = (X_test_n - muPCA) * coeff(:, 1:k95);
        k95Fold(fold) = k95;

        fprintf('PCA: %d componentes (%.0f%% de varianza) | razon n/p = %.1f\n', ...
            k95, PCA_VAR, size(X_train_pca,1) / k95);

        % --- Linea base: clasificador que predice la clase modal del
        %     conjunto de entrenamiento ---
        modalClass        = mode(double(Y_train));
        baselineAcc(fold) = sum(double(Y_test) == modalClass) / numel(Y_test);
        fprintf('Linea base (mayoritario): exactitud = %.4f\n', baselineAcc(fold));

        clsList = categories(Y_train);

        % --- Almacenamiento de las transformaciones del fold ---
        if SAVE_MODELS
            savedTransforms{fold} = struct( ...
                'mu', mu, 'sigma', sigma, ...
                'coeff', coeff(:, 1:k95), 'muPCA', muPCA, ...
                'k95', k95, 'testRep', testRepNum);
        end

        % --------------------------------------------------------
        % BUCLE DE MODELOS
        % --------------------------------------------------------
        for m = 1:nModels

            mdl = []; y_pred_test = []; scores_test = []; y_pred_train = [];

            try
                rng(SEED, 'twister');   % determinismo por modelo, relevante para
                                        % el ensamble bagged
                tStart = tic;

                switch modelNames{m}

                    case 'LDA'
                        mdl = fitcdiscr(X_train_pca, Y_train, ...
                            'DiscrimType', hp.LDA.DiscrimType, ...
                            'Gamma', hp.LDA.Gamma);

                    case 'SVM_cuadratica'
                        t = templateSVM( ...
                            'KernelFunction',  hp.SVM_cuadratica.Kernel, ...
                            'PolynomialOrder', hp.SVM_cuadratica.Order, ...
                            'BoxConstraint',   hp.SVM_cuadratica.BoxC, ...
                            'Standardize',     hp.SVM_cuadratica.Standardize);
                        mdl = fitcecoc(X_train_pca, Y_train, ...
                            'Learners', t, 'FitPosterior', true);

                    case 'Ensamble_bagged'
                        tTree = templateTree('MinLeafSize', hp.Ensamble_bagged.MinLeafSize);
                        mdl = fitcensemble(X_train_pca, Y_train, ...
                            'Method', hp.Ensamble_bagged.Method, ...
                            'NumLearningCycles', hp.Ensamble_bagged.NumLearningCycles, ...
                            'Learners', tTree);

                    otherwise
                        error('Modelo desconocido: %s', modelNames{m});
                end

                trainTime(fold, m) = toc(tStart);

                % --- Prediccion sobre prueba, con puntuaciones para ROC y PR ---
                tPred = tic;
                [y_pred_test, scores_test] = predict(mdl, X_test_pca);
                predTime(fold, m) = toc(tPred);

                % --- Prediccion sobre entrenamiento, para la brecha ---
                y_pred_train = predict(mdl, X_train_pca);

                % --- Metricas sobre prueba ---
                accTest(fold, m) = sum(y_pred_test == Y_test) / numel(Y_test);
                f1Test(fold, m)  = macroF1(Y_test, y_pred_test, clsList);

                % --- Metricas sobre entrenamiento ---
                accTrain(fold, m) = sum(y_pred_train == Y_train) / numel(Y_train);
                f1Train(fold, m)  = macroF1(Y_train, y_pred_train, clsList);

                % --- Acumulacion de las predicciones fuera de fold ---
                mn = modelNames{m};
                modelPredictions.(mn).y_true = [modelPredictions.(mn).y_true; Y_test];
                modelPredictions.(mn).y_pred = [modelPredictions.(mn).y_pred; y_pred_test];
                modelPredictions.(mn).scores = [modelPredictions.(mn).scores; scores_test];

                % --- Serializacion del modelo entrenado en forma compacta ---
                if SAVE_MODELS
                    savedModels{fold, m} = compact(mdl);
                end

                gap = accTrain(fold, m) - accTest(fold, m);
                fprintf('  %-16s Exac_ent = %.4f  Exac_pru = %.4f  brecha = %.4f | F1_pru = %.4f | t_ent = %.2f s\n', ...
                    modelLabels{m}, accTrain(fold,m), accTest(fold,m), gap, ...
                    f1Test(fold,m), trainTime(fold,m));

            catch ME
                fprintf('  %-16s ERROR: %s\n', modelLabels{m}, ME.message);
            end
        end

        nTestObs(fold) = size(X_test, 1);
        foldResults{fold} = struct( ...
            'testRep', testRepNum, ...
            'nTrain', size(X_train,1), 'nTest', size(X_test,1), ...
            'k95', k95, 'baselineAcc', baselineAcc(fold), ...
            'accTrain', accTrain(fold,:), 'accTest', accTest(fold,:), ...
            'f1Train',  f1Train(fold,:),  'f1Test',  f1Test(fold,:), ...
            'trainTime', trainTime(fold,:), 'predTime', predTime(fold,:));
    end

    % ------------------------------------------------------------
    % RESULTADOS AGREGADOS (media y desviacion estandar sobre folds)
    % ------------------------------------------------------------
    meanAccTe = mean(accTest, 1, 'omitnan');  stdAccTe = std(accTest, 0, 1, 'omitnan');
    meanF1Te  = mean(f1Test,  1, 'omitnan');  stdF1Te  = std(f1Test,  0, 1, 'omitnan');
    meanAccTr = mean(accTrain,1, 'omitnan');  stdAccTr = std(accTrain,0, 1, 'omitnan');
    meanF1Tr  = mean(f1Train, 1, 'omitnan');
    meanGap   = meanAccTr - meanAccTe;   % brecha entre entrenamiento y prueba,
                                         % empleada como indicador de sobreajuste

    fprintf('\n==================================================\n');
    fprintf('RESULTADOS AGREGADOS | blockLen = %d\n', blockLen);
    fprintf('==================================================\n');
    fprintf('Linea base mayoritaria: exactitud = %.4f +/- %.4f\n\n', ...
        mean(baselineAcc,'omitnan'), std(baselineAcc,'omitnan'));
    fprintf('%-18s %-18s %-18s %-18s %-10s\n', ...
        'Modelo','Exac_prueba','F1_prueba','Exac_entren','Brecha');
    fprintf('--------------------------------------------------------------------------------\n');
    for m = 1:nModels
        fprintf('%-18s %.4f+/-%.4f   %.4f+/-%.4f   %.4f+/-%.4f   %+.4f\n', ...
            modelLabels{m}, meanAccTe(m), stdAccTe(m), ...
            meanF1Te(m), stdF1Te(m), meanAccTr(m), stdAccTr(m), meanGap(m));
    end

    ResultsTable = table(modelLabels', meanAccTe', stdAccTe', meanF1Te', stdF1Te', ...
        meanAccTr', meanF1Tr', meanGap', ...
        'VariableNames', {'Modelo','ExacPrueba_media','ExacPrueba_desv', ...
        'F1Prueba_media','F1Prueba_desv','ExacEntren_media','F1Entren_media','Brecha_media'});

    meanTrainT = mean(trainTime,1,'omitnan');  stdTrainT = std(trainTime,0,1,'omitnan');
    meanPredT  = mean(predTime, 1,'omitnan');  stdPredT  = std(predTime, 0,1,'omitnan');
    totalPredTime = sum(predTime,1,'omitnan');
    throughput    = sum(nTestObs) ./ totalPredTime;
    TimingTable = table(modelLabels', meanTrainT', stdTrainT', ...
        meanPredT', stdPredT', throughput', ...
        'VariableNames', {'Modelo','TiempoEntren_media','TiempoEntren_desv', ...
        'TiempoPred_media','TiempoPred_desv','Rendimiento_obs_por_s'});

    % ------------------------------------------------------------
    % PALETA DE COLORES Y ESTILOS DE LINEA
    % ------------------------------------------------------------
    set(groot, 'defaultAxesFontName',  'Arial');
    set(groot, 'defaultTextFontName',  'Arial');
    set(groot, 'defaultAxesFontSize',  11);
    set(groot, 'defaultAxesLineWidth', 1.0);
    set(groot, 'defaultLineLineWidth', 1.5);

    classColors = [
        0.00, 0.45, 0.74;   % azul     - Saludable
        0.85, 0.33, 0.10;   % rojo     - Falla 1
        0.93, 0.69, 0.13;   % dorado   - Falla 2
        0.49, 0.18, 0.56;   % violeta  - Falla 3
        0.47, 0.67, 0.19;   % verde    - Falla 4
        0.30, 0.75, 0.93];  % cian     - Falla 5
    lineStyles = {'-',  '--', '-.',  ':',   '-',  '--'};
    lineWidths = [2.0,  2.0,  2.0,  2.5,   1.2,  1.2];

    figFolder = fullfile(dataFolder, sprintf('Figuras_DEFINITIVO_blockLen_%d', blockLen));
    if ~exist(figFolder, 'dir'), mkdir(figFolder); end

    % ------------------------------------------------------------
    % FIGURA 1 - COMPARACION DE MODELOS (exactitud y F1 en prueba)
    % ------------------------------------------------------------
    fig1 = figure('Name','Comparacion de modelos', 'Units','centimeters', ...
        'Position',[2 2 18 8], 'Color','w', 'PaperPositionMode','auto');

    ax1 = subplot(1,2,1);
    b1 = bar(meanAccTe, 'FaceColor','flat', 'EdgeColor','k', 'LineWidth',0.8);
    b1.CData = classColors(1:nModels, :); hold on;
    errorbar(1:nModels, meanAccTe, stdAccTe, 'k', 'LineStyle','none', ...
        'LineWidth',1.2, 'CapSize',6);
    set(ax1,'XTick',1:nModels,'XTickLabel',modelLabels,'XTickLabelRotation',30, ...
        'FontSize',10,'Box','on','LineWidth',1.0);
    ylabel('Exactitud (media \pm desv. est.)','FontSize',11);
    ylim([0 1.05]); yticks(0:0.1:1);
    grid on; set(ax1,'GridAlpha',0.25);
    text(-0.18,1.07,'(a)','Units','normalized','FontSize',12,'FontWeight','bold');

    ax2 = subplot(1,2,2);
    b2 = bar(meanF1Te, 'FaceColor','flat', 'EdgeColor','k', 'LineWidth',0.8);
    b2.CData = classColors(1:nModels, :); hold on;
    errorbar(1:nModels, meanF1Te, stdF1Te, 'k', 'LineStyle','none', ...
        'LineWidth',1.2, 'CapSize',6);
    set(ax2,'XTick',1:nModels,'XTickLabel',modelLabels,'XTickLabelRotation',30, ...
        'FontSize',10,'Box','on','LineWidth',1.0);
    ylabel('F1-macro (media \pm desv. est.)','FontSize',11);
    ylim([0 1.05]); yticks(0:0.1:1);
    grid on; set(ax2,'GridAlpha',0.25);
    text(-0.18,1.07,'(b)','Units','normalized','FontSize',12,'FontWeight','bold');

    exportgraphics(fig1, fullfile(figFolder,'fig_comparacion_modelos.png'), ...
        'Resolution',600,'BackgroundColor','white');
    exportgraphics(fig1, fullfile(figFolder,'fig_comparacion_modelos.tiff'), ...
        'Resolution',600);

    % ------------------------------------------------------------
    % FIGURA 2 - BRECHA ENTRE ENTRENAMIENTO Y PRUEBA
    % Se emplea como indicador de sobreajuste.
    % ------------------------------------------------------------
    fig2 = figure('Name','Brecha entrenamiento-prueba', 'Units','centimeters', ...
        'Position',[2 2 12 8], 'Color','w', 'PaperPositionMode','auto');
    ax = gca; hold on;
    hb = bar([meanAccTr; meanAccTe]', 'grouped', 'EdgeColor','k', 'LineWidth',0.8);
    hb(1).FaceColor = [0.60 0.60 0.60];   % entrenamiento
    hb(2).FaceColor = [0.00 0.45 0.74];   % prueba
    set(ax,'XTick',1:nModels,'XTickLabel',modelLabels,'XTickLabelRotation',20, ...
        'FontSize',10,'Box','on','LineWidth',1.0);
    ylabel('Exactitud','FontSize',11); ylim([0 1.05]); yticks(0:0.1:1);
    legend({'Entrenamiento','Prueba'},'Location','southeast','FontSize',9,'Box','on');
    grid on; set(ax,'GridAlpha',0.25);
    title(sprintf('blockLen = %d', blockLen),'FontWeight','normal','FontSize',10);
    exportgraphics(fig2, fullfile(figFolder,'fig_brecha_entren_prueba.png'), ...
        'Resolution',600,'BackgroundColor','white');
    exportgraphics(fig2, fullfile(figFolder,'fig_brecha_entren_prueba.tiff'), ...
        'Resolution',600);

    % ------------------------------------------------------------
    % FIGURAS POR MODELO
    % Matriz de confusion, curvas ROC y curvas precision-sensibilidad,
    % generadas para los tres modelos.
    % ------------------------------------------------------------
    auc_roc_all = struct(); auc_pr_all = struct();

    for r = 1:nModels
        nombreModelo = modelNames{r};
        etiquetaMod  = modelLabels{r};
        sufijo       = nombreModelo;   % diferencia los nombres de archivo

        mp = modelPredictions.(nombreModelo);
        if isempty(mp.y_true), continue; end
        y_true_all = mp.y_true;
        y_pred_all = mp.y_pred;
        scores_all = mp.scores;

        clsList2 = categories(y_true_all);
        nClasses = numel(clsList2);
        classMap = [{'Saludable'}, ...
            arrayfun(@(x) sprintf('Falla %d', x), 1:nClasses-1, 'UniformOutput', false)];

        % ---- Matriz de confusion ----
        figC = figure('Name',['Confusion - ' etiquetaMod], 'Units','centimeters', ...
            'Position',[2 2 16 14], 'Color','w', 'PaperPositionMode','auto');
        y_true_r = renamecats(y_true_all, classMap(1:nClasses));
        y_pred_r = renamecats(y_pred_all, classMap(1:nClasses));
        cm = confusionchart(y_true_r, y_pred_r);
        cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';
        cm.Title = ''; cm.FontName = 'Arial'; cm.FontSize = 10; cm.GridVisible = 'on';
        cm.DiagonalColor = [0.00 0.45 0.74]; cm.OffDiagonalColor = [0.85 0.33 0.10];
        cm.XLabel = 'Clase predicha'; cm.YLabel = 'Clase verdadera';
        exportgraphics(figC, fullfile(figFolder, ...
            sprintf('fig_matriz_confusion_%s.png', sufijo)), ...
            'Resolution',600,'BackgroundColor','white');
        exportgraphics(figC, fullfile(figFolder, ...
            sprintf('fig_matriz_confusion_%s.tiff', sufijo)),'Resolution',600);

        % ---- Curvas ROC (una contra el resto) ----
        figR = figure('Name',['ROC - ' etiquetaMod], 'Units','centimeters', ...
            'Position',[2 2 13 11], 'Color','w', 'PaperPositionMode','auto');
        axR = gca; hold on;
        auc_roc = zeros(nClasses,1);
        for i = 1:nClasses
            y_bin = (y_true_all == clsList2{i});
            [Xr, Yr, ~, A] = perfcurve(y_bin, scores_all(:,i), true);
            auc_roc(i) = A;
            plot(Xr, Yr, 'LineStyle', lineStyles{i}, 'LineWidth', lineWidths(i), ...
                'Color', classColors(i,:), ...
                'DisplayName', sprintf('%s (AUC = %.3f)', classMap{i}, A));
        end
        plot([0 1],[0 1],'Color',[0.5 0.5 0.5],'LineStyle',':', ...
            'LineWidth',1.0,'HandleVisibility','off');
        xlabel('Tasa de falsos positivos','FontSize',11);
        ylabel('Tasa de verdaderos positivos','FontSize',11);
        legend('Location','southeast','FontSize',9,'Box','on');
        xlim([0 1]); ylim([0 1.02]);
        grid on; set(axR,'Box','on','LineWidth',1.0,'GridAlpha',0.25); axis square;
        exportgraphics(figR, fullfile(figFolder, ...
            sprintf('fig_curvas_ROC_%s.png', sufijo)), ...
            'Resolution',600,'BackgroundColor','white');
        exportgraphics(figR, fullfile(figFolder, ...
            sprintf('fig_curvas_ROC_%s.tiff', sufijo)),'Resolution',600);

        % ---- Curvas precision-sensibilidad ----
        figP = figure('Name',['PR - ' etiquetaMod], 'Units','centimeters', ...
            'Position',[2 2 13 11], 'Color','w', 'PaperPositionMode','auto');
        axP = gca; hold on;
        auc_pr = zeros(nClasses,1);
        for i = 1:nClasses
            y_bin = (y_true_all == clsList2{i});
            auc_pr(i) = average_precision_score(y_bin, scores_all(:,i));
            [Rc, Pr] = perfcurve(y_bin, scores_all(:,i), true, ...
                'XCrit','reca','YCrit','prec');
            [Rc, si] = sort(Rc); Pr = Pr(si);
            v = ~isnan(Pr) & ~isnan(Rc);
            plot(Rc(v), Pr(v), 'LineStyle', lineStyles{i}, 'LineWidth', lineWidths(i), ...
                'Color', classColors(i,:), ...
                'DisplayName', sprintf('%s (PM = %.3f)', classMap{i}, auc_pr(i)));
        end
        xlabel('Sensibilidad','FontSize',11); ylabel('Precision','FontSize',11);
        legend('Location','southwest','FontSize',9,'Box','on');
        xlim([0 1]); ylim([0 1.02]);
        grid on; set(axP,'Box','on','LineWidth',1.0,'GridAlpha',0.25); axis square;
        exportgraphics(figP, fullfile(figFolder, ...
            sprintf('fig_curvas_PR_%s.png', sufijo)), ...
            'Resolution',600,'BackgroundColor','white');
        exportgraphics(figP, fullfile(figFolder, ...
            sprintf('fig_curvas_PR_%s.tiff', sufijo)),'Resolution',600);

        auc_roc_all.(nombreModelo) = auc_roc;
        auc_pr_all.(nombreModelo)  = auc_pr;

        fprintf('\nAUC y precision media para %s (blockLen = %d):\n', etiquetaMod, blockLen);
        fprintf('%-12s %-10s %-10s\n','Clase','AUC-ROC','PM');
        for i = 1:nClasses
            fprintf('%-12s %.4f     %.4f\n', classMap{i}, auc_roc(i), auc_pr(i));
        end
    end

    close all;

    % ------------------------------------------------------------
    % GUARDADO DE LA INFORMACION COMPLETA DEL ENTRENAMIENTO
    % ------------------------------------------------------------
    outputFile = fullfile(dataFolder, ...
        sprintf('LOROCV_DEFINITIVO_blockLen_%d.mat', blockLen));

    saveVars = {'accTest','f1Test','accTrain','f1Train', ...
        'trainTime','predTime','nTestObs','baselineAcc','k95Fold', ...
        'modelNames','modelLabels','foldResults', ...
        'ResultsTable','TimingTable','modelPredictions', ...
        'auc_roc_all','auc_pr_all','hp','SEED', ...
        'blockLen','step','PCA_VAR'};

    if SAVE_MODELS
        saveVars = [saveVars, {'savedModels','savedTransforms'}]; %#ok<AGROW>
    end

    save(outputFile, saveVars{:}, '-v7.3');

    fprintf('\nResultados guardados en:\n  %s\n', outputFile);
    fprintf('Figuras guardadas en:\n  %s\n', figFolder);
    fprintf('==================================================\n');
    fprintf('blockLen = %d COMPLETADO\n', blockLen);
    fprintf('==================================================\n');

    clearvars Xall ClassID RPM RepID savedModels savedTransforms modelPredictions;
end

fprintf('\n### BARRIDO DEFINITIVO COMPLETO (3 modelos, %d ventanas) ###\n', ...
    numel(blockLenList));

%% ================================================================
% FUNCIONES AUXILIARES
%% ================================================================
function f1 = macroF1(y_true, y_pred, clsList)
% Calcula el F1-macro promediando el F1 obtenido en cada clase.
    nC  = numel(clsList);
    f1c = zeros(nC,1);
    for i = 1:nC
        c  = clsList{i};
        TP = sum(y_pred == c & y_true == c);
        FP = sum(y_pred == c & y_true ~= c);
        FN = sum(y_pred ~= c & y_true == c);
        prec = TP / max(TP + FP, 1);
        rec  = TP / max(TP + FN, 1);
        f1c(i) = 2*prec*rec / max(prec + rec, eps);
    end
    f1 = mean(f1c);
end

function ap = average_precision_score(y_true, y_score)
% Calcula la precision media, entendida como el area bajo la curva
% precision-sensibilidad interpolada.
    [~, sortIdx]  = sort(y_score, 'descend');
    y_true_sorted = y_true(sortIdx);
    tp = cumsum(double(y_true_sorted));
    fp = cumsum(double(~y_true_sorted));
    precision = tp ./ (tp + fp);
    total_pos = sum(y_true);
    if total_pos == 0, ap = NaN; return; end
    recall = tp / total_pos;
    delta_recall = [recall(1); diff(recall)];
    ap = sum(delta_recall .* precision);
end
