"""superfly: baseline drone-navigation methods + the Isaac Sim / PX4 SITL
comparison harness that scores them head-to-head.

Subpackages:
    common      dependency-light shared layer (numpy/scipy/pymavlink only):
                MAVLink offboard I/O, frame conversions, sentinel files,
                the UDP depth transport.
    policies    per-method policy cores (Obs -> Cmd), each importable only
                inside that method's own venv (torch/TF live in extras).
    perception  camera-intrinsics helpers + mesh surface sampling.
    sim         Isaac Sim / Pegasus launcher + procedural obstacle fields
                (px4_sim imports Isaac at module import -- only load it
                under Isaac's interpreter).
    compare     the comparison harness: runner, method registry, metrics,
                trajectory plots. Pure numpy apart from the runner's
                subprocess calls.
"""
