import os
# Make map configuration for backup examples
map_config = {
    'geoms' : (
        ('norm_box', dict(center=(2.5, 1.0, 0.0), size=(3.0, 2.5, 2.0), p=20)),
        ('norm_box', dict(center=(-2.5, 2.5, 0.0), size=(1.25, 1.25, 2.0), p=20)),
        ('norm_box', dict(center=(-5.0, -5.0, 0.0), size=(1.875, 1.875, 2.0), p=20)),
        ('norm_box', dict(center=(5.0, -6.0, 0.0), size=(3.0, 2.0, 2.0), p=20)),
        ('norm_box', dict(center=(-7.0, 5.0, 0.0), size=(2.0, 3.0, 2.0), p=20)),
        ('norm_box', dict(center=(6.0, 7.0, 0.0), size=(3.0, 1.0, 2.0), p=20)),
        ('norm_boundary', dict(center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 2.0), p=20)),
    )}
