cmake_minimum_required(VERSION 3.8)
project(pallet_vision)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Item.msg"
  "msg/HandlingUnit.msg"
  "msg/PickerEvent.msg"
  "msg/PickerPlan.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/CountResult.msg"
  DEPENDENCIES builtin_interfaces
)

ament_python_install_package(lib)
ament_python_install_package(core)

install(PROGRAMS
  nodes/inventory_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME inventory_node
)

install(PROGRAMS
  nodes/sap_event_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sap_event_node
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
