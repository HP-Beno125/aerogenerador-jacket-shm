%% ================================================================
%  GENERAR BLOQUES + METADATA  (8 sensores, 24 canales)
%  1 estado saludable + 5 fallas  →  6 clases en total
%  ----------------------------------------------------------------
%  Variables guardadas: Xall, ClassID, RepID, RPM
%  RunID y BlockInfo eliminados (redundantes para el pipeline LOROCV)
%% ================================================================

clear; clc;

% ================================================================
% CONFIGURACIÓN
% ================================================================
dataFolder      = 'C:\Users\52953\Desktop\experimetosfinal\Datos+multiplicador';
expectedVarName = 'data';

NUM_CANALES = 24;   % 8 sensores x 3 ejes (X, Y, Z)

% --- Segmentación ---
blockLen = 50;     % muestras por ventana
step     = 50;     % igual a blockLen = sin solapamiento

% --- Mapeo índice de RPM a valor real ---
% Índice 1=10, 2=14, 3=18, 4=22 RPM
rpmMap = [10 14 18 22];

% ================================================================
% NOMBRE DE ARCHIVO DE SALIDA
% ================================================================
outputFileName = sprintf('bloques_con_metadata_%d.mat', blockLen);
outputPath     = fullfile(dataFolder, outputFileName);

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('  SEGMENTACIÓN DE DATOS PARA ML\n');
fprintf('  1 estado saludable + 5 fallas  (6 clases)\n');
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('  Sensores   : 8\n');
fprintf('  Canales    : %d  (8 sensores x 3 ejes)\n', NUM_CANALES);
fprintf('  Ventana    : %d muestras\n', blockLen);
fprintf('  Paso       : %d muestras', step);
if step < blockLen
    fprintf('  → CON SOLAPAMIENTO (%.1f%% overlap)\n', ...
        ((blockLen - step) / blockLen) * 100);
else
    fprintf('  → SIN SOLAPAMIENTO\n');
end
fprintf('  Salida     : %s\n', outputFileName);
fprintf('═══════════════════════════════════════════════════════════\n\n');

% ================================================================
% BUSCAR ARCHIVOS .MAT
% ================================================================
files = dir(fullfile(dataFolder, '*.mat'));

if isempty(files)
    error('No se encontraron archivos .mat en: %s', dataFolder);
end

fprintf('Archivos .mat encontrados: %d\n\n', numel(files));

% ================================================================
% CONTENEDORES DE BLOQUES Y METADATA
% ================================================================
Xall_parts    = cell(numel(files), 1);
ClassID_parts = cell(numel(files), 1);
RPM_parts     = cell(numel(files), 1);
RepID_parts   = cell(numel(files), 1);

validCount      = 0;
totalBlockCount = 0;

% ================================================================
% PROCESAR ARCHIVOS
% ================================================================
fprintf('Procesando archivos:\n');
fprintf('─────────────────────────────────────────────────────────────\n');

for k = 1:numel(files)

    fileName = files(k).name;
    filePath = fullfile(files(k).folder, fileName);
    [~, runName, ~] = fileparts(fileName);   % ej.: 1_2_3A

    % ----------------------------------------------------------
    % PARSEAR NOMBRE  →  formato esperado: x_z_yA
    %   x = clase (1=Healthy, 2=Fault1, …, 6=Fault5)
    %   z = repetición física (RepID)
    %   y = índice RPM (1-4)
    % ----------------------------------------------------------
    tk = regexp(runName, '^(\d+)_(\d+)_(\d+)A$', 'tokens', 'once');

    if isempty(tk)
        warning('Nombre de archivo no reconocido, omitido: %s', fileName);
        continue;
    end

    classId = str2double(tk{1});
    repId   = str2double(tk{2});
    rpmIdx  = str2double(tk{3});

    if rpmIdx < 1 || rpmIdx > numel(rpmMap)
        warning('Índice RPM fuera de rango (%d) en: %s', rpmIdx, fileName);
        continue;
    end

    rpmValue = rpmMap(rpmIdx);

    % ----------------------------------------------------------
    % CARGAR DATOS
    % ----------------------------------------------------------
    S = load(filePath);

    if isfield(S, expectedVarName)
        X = S.(expectedVarName);
    else
        vars = fieldnames(S);
        if isempty(vars)
            warning('Archivo vacío: %s', fileName);
            continue;
        end
        X = S.(vars{1});
    end

    if istable(X),    X = table2array(X);  end
    if ~isnumeric(X)
        warning('Variable no numérica en: %s', fileName);
        continue;
    end

    % ----------------------------------------------------------
    % VALIDACIONES DE DIMENSIÓN
    % ----------------------------------------------------------
    [N, C] = size(X);

    if C ~= NUM_CANALES
        warning('%s tiene %d columnas; se esperan %d (%d sensores x 3 ejes).', ...
            fileName, C, NUM_CANALES, NUM_CANALES/3);
    end

    if N < blockLen
        warning('%s: %d muestras < blockLen (%d). Archivo omitido.', ...
            fileName, N, blockLen);
        continue;
    end

    % ----------------------------------------------------------
    % SEGMENTAR EN BLOQUES
    %   Cada bloque: [blockLen x NUM_CANALES] aplanado a fila
    %   de longitud blockLen * NUM_CANALES
    % ----------------------------------------------------------
    Xblocks = makeBlocks2D(X, blockLen, step);
    nBlocks  = size(Xblocks, 1);

    % ----------------------------------------------------------
    % ACUMULAR
    % ----------------------------------------------------------
    validCount      = validCount + 1;
    totalBlockCount = totalBlockCount + nBlocks;

    Xall_parts{validCount}    = Xblocks;
    ClassID_parts{validCount} = repmat(classId,  nBlocks, 1);
    RPM_parts{validCount}     = repmat(rpmValue, nBlocks, 1);
    RepID_parts{validCount}   = repmat(repId,    nBlocks, 1);

    fprintf('  %-12s | Bloques: %4d | Clase: %d | Rep: %d | %2d RPM\n', ...
        runName, nBlocks, classId, repId, rpmValue);
end

fprintf('─────────────────────────────────────────────────────────────\n\n');

% ================================================================
% CONCATENAR
% ================================================================
if validCount == 0
    error('No se procesaron archivos válidos.');
end

Xall_parts    = Xall_parts(1:validCount);
ClassID_parts = ClassID_parts(1:validCount);
RPM_parts     = RPM_parts(1:validCount);
RepID_parts   = RepID_parts(1:validCount);

Xall    = vertcat(Xall_parts{:});
ClassID = vertcat(ClassID_parts{:});
RPM     = vertcat(RPM_parts{:});
RepID   = vertcat(RepID_parts{:});

% ================================================================
% VERIFICACIONES DE INTEGRIDAD
% ================================================================
nRows = size(Xall, 1);
nCols = size(Xall, 2);

assert(numel(ClassID) == nRows, 'Mismatch en ClassID');
assert(numel(RPM)     == nRows, 'Mismatch en RPM');
assert(numel(RepID)   == nRows, 'Mismatch en RepID');

assert(nCols == blockLen * NUM_CANALES, ...
    'Columnas esperadas: %d | Obtenidas: %d', blockLen * NUM_CANALES, nCols);

% ================================================================
% RESUMEN
% ================================================================
fprintf('PROCESAMIENTO COMPLETADO\n');
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('  Archivos procesados : %d\n', validCount);
fprintf('  Total de bloques    : %d\n', nRows);
fprintf('  Dimensión Xall      : %d x %d\n', nRows, nCols);
fprintf('    = %d bloques x (%d muestras x %d canales)\n', ...
    nRows, blockLen, NUM_CANALES);

classNamesMap = {'Healthy','Fault 1','Fault 2','Fault 3','Fault 4','Fault 5'};

fprintf('\n  Distribución de clases:\n');
for c = unique(ClassID)'
    n   = sum(ClassID == c);
    pct = n / nRows * 100;
    lbl = classNamesMap{min(c, numel(classNamesMap))};
    fprintf('    Clase %d (%s) : %5d bloques  (%.1f%%)\n', c, lbl, n, pct);
end

fprintf('\n  Distribución de RPM:\n');
for r = unique(RPM)'
    n   = sum(RPM == r);
    pct = n / nRows * 100;
    fprintf('    %2d RPM  : %5d bloques  (%.1f%%)\n', r, n, pct);
end

fprintf('\n  Distribución de repeticiones:\n');
for rep = unique(RepID)'
    n   = sum(RepID == rep);
    pct = n / nRows * 100;
    fprintf('    Rep %d   : %5d bloques  (%.1f%%)\n', rep, n, pct);
end
fprintf('═══════════════════════════════════════════════════════════\n\n');

% ================================================================
% GUARDAR
% ================================================================
fprintf('Guardando: %s\n', outputFileName);

save(outputPath, ...
    'Xall', ...        % [N x (blockLen*24)]  datos de entrada al modelo
    'ClassID', ...     % [N x 1]  etiqueta de clase  (1=Healthy, 2-6=Faults)
    'RepID', ...       % [N x 1]  repetición física  (folds LOROCV)
    'RPM', ...         % [N x 1]  velocidad de rotor (análisis por RPM)
    'blockLen', ...    % escalar  (metaparámetro de segmentación)
    'step', ...        % escalar  (idem)
    'NUM_CANALES', ... % escalar  (8 sensores x 3 ejes = 24)
    '-v7.3');

fprintf('Guardado en:\n   %s\n', outputPath);

%% ================================================================
%  FUNCIÓN AUXILIAR
%% ================================================================
function Xblocks = makeBlocks2D(X, blockLen, step)
% makeBlocks2D  Convierte [N x C] en bloques aplanados [numBloques x (blockLen*C)]
%
%   X        : señal multicanal  [N muestras x C canales]
%   blockLen : longitud de ventana (muestras)
%   step     : desplazamiento entre ventanas
%   Xblocks  : cada fila es un bloque temporal aplanado

    [N, C] = size(X);
    starts  = 1 : step : (N - blockLen + 1);
    nBlocks = numel(starts);
    Xblocks = zeros(nBlocks, blockLen * C, 'like', X);

    for i = 1:nBlocks
        idx1 = starts(i);
        idx2 = idx1 + blockLen - 1;
        Xblocks(i, :) = reshape(X(idx1:idx2, :), 1, []);
    end
end