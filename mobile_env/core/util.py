from typing import Dict

import svgpath2mpl
import matplotlib

BS_SYMBOL = (
    "M31.5,19c0-4.1-3.4-7.5-7.5-7.5s-7.5,3.4-7.5,7.5c0,2.9,1.6,5.4,4,"
    "6.7l0.9-1.8c-1.7-0.9-2.9-2.8-2.9-4.9c0-3,2.5-5.5,5.5-5.5s5.5,2.5,"
    "5.5,5.5c0,2.1-1.2,3.9-2.9,4.9l0.9,1.8C29.9,24.4,31.5,21.9,31.5,19"
    "zM37,19c0-7.2-5.8-13-13-13s-13,5.8-13,13c0,5.1,2.9,9.4,7.1,11.6l0"
    ".9-1.8c-3.6-1.8-6-5.5-6-9.8c0-6.1,4.9-11,11-11s11,4.9,11,11c0,4.3"
    "-2.5,8-6,9.8l0.9,1.8C34.1,28.4,37,24.1,37,19zM42,19c0-9.9-8.1-18-"
    "18-18S6,9.1,6,19c0,7,4.1,13.1,10,16.1l0.9-1.8C11.6,30.7,8,25.2,8,1"
    "9c0-8.8,7.2-16,16-16s16,7.2,16,16c0,6.2-3.6,11.7-8.8,14.3l0.9,1.8C"
    "37.9,32.1,42,26,42,19zM24,22c-1.7,0-3-1.3-3-3s1.3-3,3-3s3,1.3,3,3S"
    "25.7,22,24,22z M24,18c-0.6,0-1,0.4-1,1s0.4,1,1,1s1-0.4,1-1S24.6,18"
    ",24,18zM34.2,44.1L24,23.1l-10.2,21c-0.3,0.6-0.3,1.3,0.1,1.9c0.4,0."
    "6,1,0.9,1.7,0.9h16.8c0.7,0,1.3-0.4,1.7-0.9C34.5,45.5,34.5,44.7,34."
    "2,44.1z M25,29.8l2.2,4.4L25,36V29.8z M23,36l-2.2-1.8l2.2-4.4V36zM2"
    "2.5,38.2l-6,5.1l3.5-7.3L22.5,38.2zM23,40.4V45h-5.5L23,40.4z M25,40"
    ".3l5.5,4.7H25V40.3z M25.6,38.2l2.5-2.1l3.5,7.2L25.6,38.2z"
)

BS_SYMBOL = svgpath2mpl.parse_path(BS_SYMBOL)
BS_SYMBOL.vertices -= BS_SYMBOL.vertices.mean(axis=0)
# rotate (otherwise up side down): https://stackoverflow.com/a/48231144/2745116
transform = matplotlib.transforms.Affine2D().rotate_deg(180)
BS_SYMBOL = BS_SYMBOL.transformed(transform)


def deep_dict_merge(dest: Dict, source: Dict):
    """Merge dictionaries recursively (i.e. deep merge)."""
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = dest.setdefault(key, {})
            deep_dict_merge(value, node)
        else:
            dest[key] = value

    return dest
