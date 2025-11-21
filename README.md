# Laboratorio 7

## Análisis de sentimientos

## Desarrollo de un ETL 

## Exploración Tecnologica 

## MinCiencias de IA

### Detección Temprana de Plagas y Estrés Hídrico

Convocatoria MinCiencias 

Como estudiantes de la USTA proponemos un proyecto que desarrolla una solución de Inteligencia Artificial para el monitoreo agrícola en territorios rurales. El sistema utiliza nodos IoT de bajo costo con sensores ambientales y cámaras de baja resolución para detectar plagas y estrés hídrico en cultivos, enviando alertas tempranas mediante redes. 

#### Problematica Principal

Realizando un análisis dentro de Colombia encontramos que los habitantes de zonas rurales sufren perdidas de producción. Esto por detección tardía de plagas y riego ineficiente por falta de monitoreo continuo. Esto provoca baja productividad y altos costos operativos. La infraestructura de red es limitada, por lo que se requieren soluciones de bajo consumo, economía de datos y procesamiento local.

#### Obgetivos General

Desarrollar una plataforma IoT con IA embebida que permita:

1. Monitorear cultivos en tiempo real.

2. Detectar tempranamente plagas y estrés hídrico.

3. Enviar alertas a los agricultores con baja latencia.

4. Crear un dataset local reutilizable por la comunidad.

#### Flujo de Operación

- Nodo toma datos.

- Ejecuta inferencia local con TinyML.

- Si detecta plaga/estrés → envía alerta por LoRaWAN.

- Backend guarda, clasifica y notifica.

- Dashboard muestra métricas y recomendaciones.

- Datos nuevos alimentan el pipeline de reentrenamiento.
