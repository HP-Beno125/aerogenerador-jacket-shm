%% =========================================================
% VALIDACION CRUZADA DEJANDO UNA REPETICION FUERA (LOROCV)
% SONDEO PRELIMINAR DE CLASIFICADORES
%
% Se evaluan seis clasificadores de familias algoritmicas distintas
% sobre el barrido de cinco longitudes de ventana, con el fin de
% justificar de forma cuantitativa la terna retenida para el
% experimento definitivo.
%
% La configuracion es identica a la del barrido definitivo: mismos
% hiperparametros para los modelos comunes, misma semilla y mismo
% esquema de particion, de manera que las cifras de ambos analisis
% resulten directamente comparables.
%
% Se generan las tablas comparativas de exactitud y F1-macro para
% las cinco ventanas, junto con la brecha entre entrenamiento y
% prueba y los tiempos de computo.
%% =========================================================

clear; clc; close all;

% ================================================================
% CONFIGURACION GLOBAL
% ================================================================
dataFolder   = 'C:\Users\52953\Desktop\experimetosfinal\REsultados para entrega de tesis\Boques_preliminar_sin_multiplicador';
inputPattern = 'bloques_con_metadata_%d.mat';

blockLenList = [50, 100, 125, 196, 245];

SEED    = 42;    % semilla global para reproducibilidad exacta
PCA_VAR = 95;    % umbral de varianza retenida por el PCA

% --- Modelos evaluados en el sondeo ---
modelNames  = {'LDA', 'SVM_cuadratica', 'SVM_gaussiana', ...
               'Arbol_fino', 'KNN_coseno', 'Ensamble_bagged'};
modelLabels = {'LDA', 'SVM cuadratica', 'SVM gaussiana', ...
               'Arbol fino', 'KNN coseno', 'Ensamble bagged'};
nModels     = numel(modelNames);

% --- Hiperparametros explicitos de los seis modelos ---
hp = struct();

hp.LDA.DiscrimType = 'linear';
hp.LDA.Gamma       = 0;

hp.SVM_cuadratica.Kernel      = 'polynomial';
hp.SVM_cuadratica.Order       = 2;
hp.SVM_cuadratica.BoxC        = 1;
hp.SVM_cuadratica.Standardize = false;

hp.SVM_gaussiana.Kernel      = 'gaussian';
hp.SVM_gaussiana.BoxC        = 1;
hp.SVM_gaussiana.KernelScale = 1;
hp.SVM_gaussiana.Standardize = false;

hp.Arbol_fino.MaxNumSplits = 100;
hp.Arbol_fino.SplitCriterion = 'gdi';
hp.Arbol_fino.MinLeafSize    = 1;

hp.KNN_coseno.NumNeighbors = 10;
hp.KNN_coseno.Distance     = 'cosine';
hp.KNN_coseno.DistanceWeight = 'equal';

hp.Ensamble_bagged.Method            = 'Bag';
hp.Ensamble_bagged.NumLearningCycles = 50;
hp.Ensamble_bagged.MinLeafSize       = 1;

% ================================================================
% ACUMULADORES GLOBALES (modelos x ventanas)
% ================================================================
nWin = numel(blockLenList);
accTe_media  = nan(nModels, nWin);   accTe_desv  = nan(nModels, nWin);
f1Te_media   = nan(nModels, nWin);   f1Te_desv   = nan(nModels, nWin);
accTr_media  = nan(nModels, nWin);   brecha      = nan(nModels, nWin);
tEntren      = nan(nModels, nWin);   tPred       = nan(nModels, nWin);
baseline_win = nan(1, nWin);         k95_win     = nan(1, nWin);

% ================================================================
% BUCLE EXTERNO SOBRE LAS LONGITUDES DE VENTANA
% ================================================================
for w = 1:nWin

    bl = blockLenList(w);
    rng(SEED, 'twister');

    inputFile = fullfile(dataFolder, sprintf(inputPattern, bl));
    if ~isfile(inputFile)
        warning('No se encontro %s. Se omite blockLen = %d.', inputFile, bl);
        continue;
    end

    load(inputFile, 'Xall', 'ClassID', 'RPM', 'RepID', 'blockLen', 'step');

    fprintf('\n==================================================\n');
    fprintf('SONDEO PRELIMINAR - 6 modelos | blockLen = %d\n', blockLen);
    fprintf('==================================================\n');
    fprintf('Conjunto de datos: %d bloques\n', size(Xall,1));

    % ------------------------------------------------------------
    % IDENTIFICACION DE LAS REPETICIONES FISICAS
    % ------------------------------------------------------------
    PhysRepID  = strcat("C", string(ClassID), "_R", string(RepID));
    [G, ~]     = findgroups(PhysRepID);
    physClass  = splitapply(@(x) x(1), ClassID, G);
    physRepNum = splitapply(@(x) x(1), RepID,   G);

    uniqueRepNumbers = unique(physRepNum);
    nFolds = numel(uniqueRepNumbers);
    fprintf('Clases: %d | folds (repeticiones): %d\n\n', ...
        numel(unique(physClass)), nFolds);

    % ------------------------------------------------------------
    % ESTRUCTURAS DE ALMACENAMIENTO POR VENTANA
    % ------------------------------------------------------------
    accTest   = nan(nFolds, nModels);
    f1Test    = nan(nFolds, nModels);
    accTrain  = nan(nFolds, nModels);
    f1Train   = nan(nFolds, nModels);
    trainTime = nan(nFolds, nModels);
    predTime  = nan(nFolds, nModels);
    nTestObs  = zeros(nFolds, 1);
    baselineAcc = nan(nFolds, 1);
    k95Fold     = zeros(nFolds, 1);
    foldResults = cell(nFolds, 1);

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

        % --- Estandarizacion z-score, ajustada solo con entrenamiento ---
        mu    = mean(X_train, 1);
        sigma = std(X_train, 0, 1);
        sigma(sigma == 0) = 1;
        X_train_n = (X_train - mu) ./ sigma;
        X_test_n  = (X_test  - mu) ./ sigma;

        % --- PCA, ajustado solo con entrenamiento ---
        warnState = warning('off', 'stats:pca:ColRankDefX');
        [coeff, scoreTrain, ~, ~, explained, muPCA] = pca(X_train_n);
        warning(warnState);
        cumExplained = cumsum(explained);
        k95 = find(cumExplained >= PCA_VAR, 1, 'first');
        X_train_pca = scoreTrain(:, 1:k95);
        X_test_pca  = (X_test_n - muPCA) * coeff(:, 1:k95);
        k95Fold(fold) = k95;

        fprintf('Bloques: %d entren. / %d prueba | PCA: %d componentes | n/p = %.1f\n', ...
            size(X_train,1), size(X_test,1), k95, size(X_train_pca,1)/k95);

        % --- Linea base del clasificador mayoritario ---
        modalClass        = mode(double(Y_train));
        baselineAcc(fold) = sum(double(Y_test) == modalClass) / numel(Y_test);

        clsList = categories(Y_train);

        % --------------------------------------------------------
        % BUCLE DE MODELOS
        % --------------------------------------------------------
        for m = 1:nModels

            mdl = [];

            try
                rng(SEED, 'twister');   % determinismo por modelo
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

                    case 'SVM_gaussiana'
                        t = templateSVM( ...
                            'KernelFunction', hp.SVM_gaussiana.Kernel, ...
                            'BoxConstraint',  hp.SVM_gaussiana.BoxC, ...
                            'KernelScale',    hp.SVM_gaussiana.KernelScale, ...
                            'Standardize',    hp.SVM_gaussiana.Standardize);
                        mdl = fitcecoc(X_train_pca, Y_train, ...
                            'Learners', t, 'FitPosterior', true);

                    case 'Arbol_fino'
                        mdl = fitctree(X_train_pca, Y_train, ...
                            'MaxNumSplits',   hp.Arbol_fino.MaxNumSplits, ...
                            'SplitCriterion', hp.Arbol_fino.SplitCriterion, ...
                            'MinLeafSize',    hp.Arbol_fino.MinLeafSize);

                    case 'KNN_coseno'
                        mdl = fitcknn(X_train_pca, Y_train, ...
                            'NumNeighbors',   hp.KNN_coseno.NumNeighbors, ...
                            'Distance',       hp.KNN_coseno.Distance, ...
                            'DistanceWeight', hp.KNN_coseno.DistanceWeight);

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

                tP = tic;
                y_pred_test = predict(mdl, X_test_pca);
                predTime(fold, m) = toc(tP);

                y_pred_train = predict(mdl, X_train_pca);

                accTest(fold, m)  = sum(y_pred_test == Y_test) / numel(Y_test);
                f1Test(fold, m)   = macroF1(Y_test, y_pred_test, clsList);
                accTrain(fold, m) = sum(y_pred_train == Y_train) / numel(Y_train);
                f1Train(fold, m)  = macroF1(Y_train, y_pred_train, clsList);

                gap = accTrain(fold, m) - accTest(fold, m);
                fprintf('  %-16s Exac_ent = %.4f  Exac_pru = %.4f  brecha = %+.4f | F1_pru = %.4f | t_ent = %.2f s\n', ...
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
    % AGREGACION DE LA VENTANA
    % ------------------------------------------------------------
    accTe_media(:,w) = mean(accTest, 1, 'omitnan')';
    accTe_desv(:,w)  = std(accTest,  0, 1, 'omitnan')';
    f1Te_media(:,w)  = mean(f1Test,  1, 'omitnan')';
    f1Te_desv(:,w)   = std(f1Test,   0, 1, 'omitnan')';
    accTr_media(:,w) = mean(accTrain,1, 'omitnan')';
    brecha(:,w)      = accTr_media(:,w) - accTe_media(:,w);
    tEntren(:,w)     = mean(trainTime,1,'omitnan')';
    tPred(:,w)       = mean(predTime, 1,'omitnan')';
    baseline_win(w)  = mean(baselineAcc,'omitnan');
    k95_win(w)       = mean(k95Fold);

    fprintf('\n--- Resumen blockLen = %d | linea base = %.4f ---\n', ...
        blockLen, baseline_win(w));
    for m = 1:nModels
        fprintf('%-16s Exac = %.4f +/- %.4f | F1 = %.4f +/- %.4f | brecha = %+.4f\n', ...
            modelLabels{m}, accTe_media(m,w), accTe_desv(m,w), ...
            f1Te_media(m,w), f1Te_desv(m,w), brecha(m,w));
    end

    clearvars Xall ClassID RPM RepID;
end

% ================================================================
% TABLAS COMPARATIVAS FINALES
% ================================================================
encabezado = arrayfun(@(x) sprintf('n=%d', x), blockLenList, ...
    'UniformOutput', false);

fprintf('\n\n==================================================\n');
fprintf('TABLA 1 - EXACTITUD (media +/- desviacion estandar)\n');
fprintf('==================================================\n');
imprimirTabla(modelLabels, encabezado, accTe_media, accTe_desv);

fprintf('\n==================================================\n');
fprintf('TABLA 2 - F1-MACRO (media +/- desviacion estandar)\n');
fprintf('==================================================\n');
imprimirTabla(modelLabels, encabezado, f1Te_media, f1Te_desv);

fprintf('\n==================================================\n');
fprintf('TABLA 3 - BRECHA ENTRE ENTRENAMIENTO Y PRUEBA\n');
fprintf('==================================================\n');
fprintf('%-18s', 'Modelo');
fprintf('%12s', encabezado{:}); fprintf('\n');
fprintf('%s\n', repmat('-', 1, 18 + 12*nWin));
for m = 1:nModels
    fprintf('%-18s', modelLabels{m});
    fprintf('%12.4f', brecha(m,:)); fprintf('\n');
end

fprintf('\n==================================================\n');
fprintf('TABLA 4 - TIEMPO MEDIO DE ENTRENAMIENTO POR FOLD [s]\n');
fprintf('==================================================\n');
fprintf('%-18s', 'Modelo');
fprintf('%12s', encabezado{:}); fprintf('\n');
fprintf('%s\n', repmat('-', 1, 18 + 12*nWin));
for m = 1:nModels
    fprintf('%-18s', modelLabels{m});
    fprintf('%12.2f', tEntren(m,:)); fprintf('\n');
end

fprintf('\n--- Referencias ---\n');
fprintf('Linea base por ventana : ');
fprintf('%.4f  ', baseline_win); fprintf('\n');
fprintf('Componentes PCA (media): ');
fprintf('%.1f  ', k95_win); fprintf('\n');

% ================================================================
% GUARDADO
% ================================================================
outputFile = fullfile(dataFolder, 'LOROCV_SONDEO_6modelos.mat');
save(outputFile, 'accTe_media', 'accTe_desv', 'f1Te_media', 'f1Te_desv', ...
    'accTr_media', 'brecha', 'tEntren', 'tPred', ...
    'baseline_win', 'k95_win', 'modelNames', 'modelLabels', ...
    'blockLenList', 'hp', 'SEED', 'PCA_VAR', '-v7.3');

fprintf('\nResultados guardados en:\n  %s\n', outputFile);
fprintf('\n### SONDEO COMPLETO (6 modelos, %d ventanas) ###\n', nWin);

%% ================================================================
% FUNCIONES AUXILIARES
%% ================================================================
function imprimirTabla(filas, columnas, medias, desviaciones)
% Imprime en consola una tabla de medias con su desviacion estandar.
    nW = numel(columnas);
    fprintf('%-18s', 'Modelo');
    fprintf('%18s', columnas{:}); fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 18 + 18*nW));
    for m = 1:numel(filas)
        fprintf('%-18s', filas{m});
        for w = 1:nW
            fprintf('%11.3f +/-%.3f', medias(m,w), desviaciones(m,w));
        end
        fprintf('\n');
    end
end

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