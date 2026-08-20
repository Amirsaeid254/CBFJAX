"""
Map configuration for the C-ADP ground-robot example.

Obstacle field taken from the flowbarrier unicycle example: 46 cylindrical
obstacles inside a 10 x 10 m norm boundary, standing in for the map of the
C-ADP paper (which does not publish its obstacle coordinates).

No 'velocity' entry: the speed and angular-velocity limits of the paper are the
quadratic barriers sbar^2 - s^2 and omegabar^2 - omega^2, which the example
builds as separate 'func' entries rather than as box barriers.
"""

map_config = {
    'geoms': (
        ('cylinder', {'center': (6.2130, 5.8929), 'radius': 0.5771}),
        ('cylinder', {'center': (1.7019, -3.4985), 'radius': 0.8785}),
        ('cylinder', {'center': (6.7265, 4.0497), 'radius': 0.9942}),
        ('cylinder', {'center': (3.3322, 1.1502), 'radius': 0.9823}),
        ('cylinder', {'center': (-0.6651, -2.5190), 'radius': 0.4553}),
        ('cylinder', {'center': (2.3309, -1.2814), 'radius': 0.7491}),
        ('cylinder', {'center': (4.8359, 4.8576), 'radius': 1.0982}),
        ('cylinder', {'center': (6.1753, 0.3786), 'radius': 0.4345}),
        ('cylinder', {'center': (-1.6906, 5.4668), 'radius': 0.6173}),
        ('cylinder', {'center': (-6.2984, 3.7253), 'radius': 1.4128}),
        ('cylinder', {'center': (6.2437, 3.7842), 'radius': 1.4669}),
        ('cylinder', {'center': (0.2376, 4.0153), 'radius': 0.7969}),
        ('cylinder', {'center': (2.9078, -4.6220), 'radius': 0.9473}),
        ('cylinder', {'center': (-5.4113, -6.8586), 'radius': 0.3424}),
        ('cylinder', {'center': (-4.5174, -5.1970), 'radius': 0.9816}),
        ('cylinder', {'center': (-3.3882, 6.3649), 'radius': 1.4188}),
        ('cylinder', {'center': (4.6121, -3.6274), 'radius': 0.5230}),
        ('cylinder', {'center': (1.1229, 3.6693), 'radius': 0.6494}),
        ('cylinder', {'center': (-4.3392, 6.5532), 'radius': 0.7181}),
        ('cylinder', {'center': (0.3081, 0.6679), 'radius': 1.0175}),
        ('cylinder', {'center': (4.7827, 2.9601), 'radius': 0.7830}),
        ('cylinder', {'center': (-4.3347, 5.6848), 'radius': 0.9477}),
        ('cylinder', {'center': (-2.5188, 4.5451), 'radius': 1.3531}),
        ('cylinder', {'center': (1.7506, -0.5856), 'radius': 0.4919}),
        ('cylinder', {'center': (-0.6233, -4.2486), 'radius': 0.7440}),
        ('cylinder', {'center': (-5.3258, -0.1985), 'radius': 0.6980}),
        ('cylinder', {'center': (0.8580, 3.0467), 'radius': 1.1832}),
        ('cylinder', {'center': (2.1390, -4.6164), 'radius': 0.8798}),
        ('cylinder', {'center': (-0.4618, 3.1852), 'radius': 0.5000}),
        ('cylinder', {'center': (-7.6589, 2.5671), 'radius': 1.1000}),
        ('cylinder', {'center': (-9.0354, -1.5000), 'radius': 0.9000}),
        ('cylinder', {'center': (8.2000, -1.0000), 'radius': 1.0000}),
        ('cylinder', {'center': (8.6233, 2.0315), 'radius': 0.9000}),
        ('cylinder', {'center': (7.8000, -5.0000), 'radius': 0.8500}),
        ('cylinder', {'center': (-1.5000, -6.8000), 'radius': 0.9000}),
        ('cylinder', {'center': (0.8000, -6.0000), 'radius': 0.7500}),
        ('cylinder', {'center': (4.5000, -6.5000), 'radius': 1.0000}),
        ('cylinder', {'center': (-0.5000, 7.0000), 'radius': 0.7000}),
        ('cylinder', {'center': (6.5000, -2.0000), 'radius': 0.7000}),
        ('cylinder', {'center': (-7.9333, -2.2333), 'radius': 0.7500}),
        ('cylinder', {'center': (3.5000, 6.5000), 'radius': 0.6500}),
        ('norm_boundary', {'center': (0.0000, 0.0000), 'size': (10.0000, 10.0000)}),
        ('cylinder', {'center': (-8.3000, -5.6000), 'radius': 1.0000}),
        ('cylinder', {'center': (9.2375, -1.7354), 'radius': 0.7000}),
        ('cylinder', {'center': (-4.7249, -1.0143), 'radius': 0.7500}),
        ('cylinder', {'center': (-4.3233, 0.2268), 'radius': 0.7500}),
        ('cylinder', {'center': (-2.6766, 1.3647), 'radius': 0.4553}),
    ),
}
