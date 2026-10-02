# aerogenerador-jacket-shm
Código de adquisición, procesamiento y clasificación para el monitoreo de salud estructural de un aerogenerador offshore tipo jacket a escala.


## Contenido

- `firmware/` Firmware del nodo ESP32-C3 con acelerómetro ADXL355.
- `python/` Aplicación receptora: sincroniza relojes, coordina capturas
  y almacena los registros de los ocho nodos.
- `matlab/` Procesamiento y clasificación.

## Orden de ejecución

1. `Ventanas_deslizantes_Meta_datos.m` segmenta los registros en bloques
   y genera los metadatos de clase, repetición y velocidad.
2. `LOROCV_SONDEO_6modelos.m` compara seis clasificadores sobre cinco
   longitudes de ventana.
3. `LOROCV_DEFINITIVO.m` evalúa los tres modelos retenidos y genera
   matrices de confusión, curvas ROC y curvas precisión-sensibilidad.

