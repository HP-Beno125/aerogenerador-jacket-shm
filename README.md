# Monitoreo de salud estructural de un aerogenerador offshore tipo jacket


## Archivos

| Archivo | Descripción |
|---|---|
| `ADXL355_nodo_emisor.ino` | Firmware del nodo ESP32-C3 con acelerómetro ADXL355. Configura el sensor, captura los tres ejes y transmite por TCP/IP. |
| `Servidor_VF.py` | Aplicación receptora. Sincroniza los relojes de los ocho nodos, coordina el inicio de las capturas y almacena los registros. |
| `Ventanas_deslizantes_Meta_datos.m` | Segmenta los registros en bloques sin solapamiento y genera las etiquetas de condición, repetición física y velocidad de giro. |
| `LOROCV_6_modelos.m` | Compara seis clasificadores sobre cinco longitudes de ventana, bajo validación cruzada dejando una repetición fuera. |
| `LOROCV_DEFINITIVO.m` | Evalúa los tres modelos retenidos y genera matrices de confusión, curvas ROC y curvas precisión-sensibilidad. |

## Orden de ejecución

Los scripts de MATLAB se corren en el orden en que aparecen en la tabla.
El primero produce los archivos `.mat` que consumen los otros dos.
