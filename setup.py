from setuptools import find_packages, setup

package_name = "pallet_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pallet Vision Team",
    maintainer_email="vision@cencosud.com",
    description="Pallet Vision with monolithic inventory node",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "inventory_node = nodes.inventory_node:main",
            "sap_event_node = nodes.sap_event_node:main",
        ],
    },
)
