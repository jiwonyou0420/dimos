"""Hackathon blueprints for Unitree GO2."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "unitree_go2_find_object": ["unitree_go2_find_object"],
        "unitree_go2_hackathon_hybrid": ["unitree_go2_hackathon_hybrid"],
    },
)
