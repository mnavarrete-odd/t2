# new-pallets-vision

Proyecto ROS2 limpio para Pallet Vision, sin sufijos `v2`.

## Nodos
- `inventory_node`: nodo monolítico con detector + photographer + counter para cámara primaria/secundaria.
- `sap_event_node`: API HTTP simple (`POST /api/signal`) que publica `ITEM_START`, `ITEM_END`, `ORDER_END` en `/picker/events`.

## Flujo
1. `ITEM_START`: fuerza KF inicial por cámara y activa photographer.
2. Tarea activa: frames -> detección -> photographer -> keyframes -> counter.
3. `ITEM_END`: fuerza KF final y desactiva photographer.
4. `ORDER_END`: cierra tarea si aplica, publica resumen final y resetea estado.

## Launch
```bash
ros2 launch pallet_vision pallet_vision.launch.py
```

Simulación:
```bash
ros2 launch pallet_vision pallet_vision_sim.launch.py
```

## Config
- `config/inventory_node_config.yaml`
- `config/sap_event_config.yaml`
- `config/counter_default.yaml`

## Tests
```bash
python3 -m pytest -q tests/test_event_manager.py tests/test_boundary_keyframes.py tests/test_counter_adapter.py tests/test_integration_signal_cycle.py
```

## Guía completa de instalación/uso
- `docs/GUIA_USO_ROS2.md`
