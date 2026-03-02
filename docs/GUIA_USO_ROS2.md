# Guía de instalación y uso (PC nuevo)

Esta guía sirve para ejecutar `new-pallets-vision` en otro PC con ROS2.

## 1. Requisitos

- Ubuntu con ROS2 instalado (recomendado: Humble en 22.04 o Jazzy en 24.04).
- Python 3.10+.
- `colcon`, `cv_bridge`, `message_filters`.
- Drivers/nodos de cámara publicando:
  - `/primary_camera/color/image_raw`
  - `/primary_camera/depth/image_raw`
  - `/secondary_camera/color/image_raw`
  - `/secondary_camera/depth/image_raw`

## 2. Instalar ROS2 (si aún no está instalado)

Si ya tienes ROS2 funcionando, salta a la sección 3.

Referencia recomendada: instalación oficial de ROS2 para tu distro de Ubuntu.

Validación rápida:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 --version
```

Si `ros2` responde correctamente, continúa.

## 3. Crear workspace y copiar el proyecto

```bash
mkdir -p ~/ws_pallet/src
cd ~/ws_pallet/src
```

Copia la carpeta del proyecto como paquete `pallet_vision`:

```bash
cp -r /ruta/al/proyecto/new-pallets-vision ./pallet_vision
```

Estructura esperada:

```bash
~/ws_pallet/src/pallet_vision
```

## 4. Instalar dependencias Python del proyecto

```bash
cd ~/ws_pallet/src/pallet_vision
python3 -m pip install -r requirements.txt
```
ros2 run multicam simulate_camera --fps 1 --folder "~/Oddness/Cencosud/cenco_ws/data/data_captured_26_02/output/pickeo_26_02"
## 5. Compilar

```bash
cd ~/ws_pallet
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select pallet_vision
```

## 6. Cargar entorno

En cada terminal nueva:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/ws_pallet/install/setup.bash
```

## 7. Ejecutar Pallet Vision

### Opción A: launcher completo (recomendado)

```bash
ros2 launch pallet_vision pallet_vision.launch.py
```

Este launcher levanta:

- `inventory_node`
- `sap_event_node`

### Opción B: modo simulación

```bash
ros2 launch pallet_vision pallet_vision_sim.launch.py
```

## 8. Enviar señales SAP por curl

Endpoint por defecto:

- `http://<IP_DEL_PC>:8080/api/signal`

### ITEM_START

```bash
curl -X POST http://127.0.0.1:8080/api/signal \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "ITEM_START",
    "order_id": "ORDER-1001",
    "hu_id": "HU-001",
    "sku": "SKU-ABC",
    "quantity": 2,
    "description": "Producto prueba"
  }'
```

### ITEM_END

```bash
curl -X POST http://127.0.0.1:8080/api/signal \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "ITEM_END",
    "order_id": "ORDER-1001",
    "hu_id": "HU-001"
  }'
```

### ORDER_END

```bash
curl -X POST http://127.0.0.1:8080/api/signal \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "ORDER_END",
    "order_id": "ORDER-1001"
  }'
```

## 9. Verificar salida en ROS

### Eventos recibidos

```bash
ros2 topic echo /picker/events
```

### Conteo por cámara

```bash
ros2 topic echo /inventory/count/primary
ros2 topic echo /inventory/count/secondary
```

### Estado general y resultado de orden

```bash
ros2 topic echo /inventory/state
ros2 topic echo /inventory/order_result
```

## 10. Configuración útil

Archivo principal:

- `config/inventory_node_config.yaml`

Parámetros claves:

- `detector_mode`: `shared` o `per_camera`
- `detection_mode`: `realtime` o `no_drop`
- `debug.save_kfs`: guardar keyframes
- `debug.save_only_boundary_kfs`: solo KFs de borde
- `debug.save_counter_frames`: guardar frames del contador
- `debug.save_counter_video`: video de tracking

Config SAP:

- `config/sap_event_config.yaml`

Formato del YAML de SAP (ROS2 params):

```yaml
sap_event_node:
  ros__parameters:
    host: "0.0.0.0"
    port: 8080
    api_endpoint: "/api/signal"
    auth_enabled: false
    auth_api_key: ""
    max_request_size: 10485760
```

## 11. Autenticación de API (opcional)

En `config/sap_event_config.yaml`:

```yaml
auth:
  enabled: true
  api_key: "MI_API_KEY"
```

Llamada curl con API key:

```bash
curl -X POST http://127.0.0.1:8080/api/signal \
  -H "Content-Type: application/json" \
  -H "X-API-Key: MI_API_KEY" \
  -d '{"event_type":"ORDER_END","order_id":"ORDER-1001"}'
```

## 12. Problemas comunes

- `ModuleNotFoundError`: faltan dependencias Python (`pip install -r requirements.txt`).
- No llegan imágenes: revisar topics de cámara y nombres en config.
- Sin conteo en `/inventory/count/*`: revisar detecciones y que exista `ITEM_START` activo.
- Error de modelo YOLO: validar `detector_model_path` y archivo en `models/`.

## 13. Error típico en PC nuevo y solución rápida

Si aparece:

- `Cannot have a value before ros__parameters` en `sap_event_config.yaml`

Entonces el archivo no está en formato ROS2. Usa exactamente el formato de la sección 10.

Si aparece:

- `FileNotFoundError ... models/product_detector.pt`

Entonces falta el modelo en el PC destino. Soluciones:

1. Copiar el modelo a `src/pallet_vision/models/product_detector.pt` y recompilar.
2. O poner ruta absoluta en `detector_model_path` dentro de `inventory_node_config.yaml`.
