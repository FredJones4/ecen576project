import os, pkg_resources
from pyanatomogram import Anatomogram

svg_path = pkg_resources.resource_filename(
    'pyanatomogram',
    'anatomogram/src/svg/homo_sapiens.svg'
)
print(svg_path, os.path.exists(svg_path))
