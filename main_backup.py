"""
Reflector Position Optimization - L-Shaped Corridor

This script implements physics-aware optimal placement for mechanical reflectors 
in NLOS (Non-Line-of-Sight) scenarios using Sionna RT.

Goal: Find optimal deployment position for a mechanical reflector using gradient descent
Scenario: L-shaped corridor where direct LOS between Tx and Rx is blocked
"""

import numpy as np
import tensorflow as tf
import sionna
from sionna.rt import load_scene
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

class ReflectorController:
    """
    Helper class for controlling a flat reflector's position and orientation.
    
    This class provides a convenient interface for:
    - Setting position (x, y, z) in meters
    - Setting orientation using Euler angles (α, β, γ) in radians
    - Getting/setting parameters as vectors for optimization
    """
    
    def __init__(self, reflector):
        """
        Args:
            reflector: sionna.rt.SceneObject representing the reflector
        """
        self.reflector = reflector
        
    def set_position(self, x, y, z):
        """Set reflector position in meters"""
        pos = [float(x), float(y), float(z)]
        self.reflector.position = pos
        
    def get_position(self):
        """Get reflector position as numpy array"""
        pos = self.reflector.position[0]
        return np.array([pos[0], pos[1], pos[2]])
        
    def set_orientation(self, alpha, beta, gamma):
        """
        Set reflector orientation using Euler angles.
        
        Args:
            alpha: Rotation around z-axis (yaw) in radians
            beta: Rotation around y-axis (pitch) in radians
            gamma: Rotation around x-axis (roll) in radians
        """
        self.reflector.orientation = [float(alpha), float(beta), float(gamma)]
        
    def get_orientation(self):
        """Get reflector orientation as numpy array [alpha, beta, gamma]"""
        orient = self.reflector.orientation[0]
        return np.array([orient[0], orient[1], orient[2]])
    
    def set_params(self, params):
        """
        Set position and orientation from parameter vector.
        
        Args:
            params: Array of [x, y, z, alpha, beta, gamma]
        """
        assert len(params) == 6, "params must have 6 elements: [x, y, z, alpha, beta, gamma]"
        self.set_position(params[0], params[1], params[2])
        self.set_orientation(params[3], params[4], params[5])
        
    def get_params(self):
        """
        Get position and orientation as parameter vector.
        
        Returns:
            Array of [x, y, z, alpha, beta, gamma]
        """
        pos = self.get_position()
        orient = self.get_orientation()
        return np.concatenate([pos, orient])
    
    def __repr__(self):
        pos = self.get_position()
        orient = self.get_orientation()
        return (f"ReflectorController(\n"
                f"  position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m\n"
                f"  orientation: [{np.degrees(orient[0]):.1f}°, "
                f"{np.degrees(orient[1]):.1f}°, {np.degrees(orient[2]):.1f}°]\n"
                f")")


def create_flat_reflector_mesh(width=2.0, height=2.0):
    """
    Creates a flat rectangular reflector mesh centered at origin.
    
    Args:
        width: Width of reflector in meters (along y-axis)
        height: Height of reflector in meters (along z-axis)
    
    Returns:
        Mitsuba mesh object
    """
    # Define vertices for a rectangle in the y-z plane (normal along x-axis)
    # This orientation makes it easier to position along walls
    w2, h2 = width / 2, height / 2
    vertices = np.array([
        [-0.01, -w2, -h2],  # Bottom-left
        [-0.01,  w2, -h2],  # Bottom-right  
        [-0.01,  w2,  h2],  # Top-right
        [-0.01, -w2,  h2],  # Top-left
    ], dtype=np.float32)
    
    # Define two triangular faces (rectangle = 2 triangles)
    faces = np.array([
        [0, 1, 2],  # First triangle
        [0, 2, 3],  # Second triangle
    ], dtype=np.uint32)
    
    # Create Mitsuba mesh
    mesh = mi.Mesh(
        "reflector_mesh",
        vertex_count=len(vertices),
        face_count=len(faces),
        has_vertex_normals=False,
        has_vertex_texcoords=False
    )
    
    # Set mesh data
    mesh_params = mi.traverse(mesh)
    # Transpose vertices to shape (3, N) as required by mi.Point3f
    mesh_params['vertex_positions'] = dr.ravel(mi.Point3f(vertices.T))
    mesh_params['faces'] = dr.ravel(mi.Vector3u(faces.T))
    mesh_params.update()
    
    return mesh


def setup_scene(scene_path):
    """
    Load scene and add transmitter and receiver.
    
    Args:
        scene_path: Path to the scene XML file
        
    Returns:
        tuple: (scene, tx, rx)
    """
    # Load the L-shaped corridor scene
    scene = load_scene(scene_path)
    
    # Add Transmitter (Tx) - Deep in the West Leg
    # Position: ceiling mount at x=8, y=13, z=2.7
    tx = sionna.rt.Transmitter(
        name="Tx",
        position=[8, 13, 2.7],
        orientation=[0, 0, 0]
    )
    scene.add(tx)
    
    # Add Receiver (Rx) - Deep in the North Leg
    # Position: user height at x=16, y=6.5, z=1.5
    rx = sionna.rt.Receiver(
        name="Rx",
        position=[16, 6.5, 1.5],
        orientation=[0, 0, 0]
    )
    scene.add(rx)
    
    print(f"✓ Scene loaded: {scene_path}")
    print(f"  Tx position: {tx.position}")
    print(f"  Rx position: {rx.position}")
    
    return scene, tx, rx


def create_reflector(scene, width=2.0, height=2.0):
    """
    Create a controllable flat reflector and add it to the scene.
    
    Args:
        scene: Sionna RT Scene object
        width: Width of reflector in meters
        height: Height of reflector in meters
        
    Returns:
        tuple: (reflector, reflector_ctrl)
    """
    # Create a highly reflective metal material
    reflector_material = sionna.rt.ITURadioMaterial(
        name="reflector_metal",
        itu_type="metal",
        thickness=0.002,  # 2mm thick metal plate
    )
    
    # Create the reflector mesh
    reflector_mesh = create_flat_reflector_mesh(width=width, height=height)
    
    # Create SceneObject from the mesh
    reflector = sionna.rt.SceneObject(
        mi_mesh=reflector_mesh,
        name="reflector",
        radio_material=reflector_material
    )
    
    # Add reflector to scene
    scene.edit(add=reflector)
    
    # Create controller for the reflector
    reflector_ctrl = ReflectorController(reflector)
    
    print(f"✓ Created flat reflector: {reflector.name}")
    print(f"  - Size: {width}m x {height}m")
    print(f"  - Material: {reflector.radio_material.name}")
    
    return reflector, reflector_ctrl


def set_reflector_orientation(reflector, alpha_deg, beta_deg, gamma_deg):
    """
    Set reflector orientation using angles in degrees.
    
    Args:
        reflector: SceneObject representing the reflector
        alpha_deg: Rotation around z-axis (yaw) in degrees
        beta_deg: Rotation around y-axis (pitch) in degrees  
        gamma_deg: Rotation around x-axis (roll) in degrees
        
    Returns:
        Current orientation in radians
    """
    alpha_rad = float(np.radians(alpha_deg))
    beta_rad = float(np.radians(beta_deg))
    gamma_rad = float(np.radians(gamma_deg))
    
    reflector.orientation = [alpha_rad, beta_rad, gamma_rad]
    
    return reflector.orientation


def main() -> None:
    """Main entry point for reflector position optimization."""
    
    print("=" * 60)
    print("Reflector Position Optimization - L-Shaped Corridor")
    print("=" * 60)
    
    # Print versions
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Sionna version: {sionna.__version__}")
    print()
    
    # Setup scene with transmitter and receiver
    scene_path = "/home/hieule/blender/models/hallway-L.xml"
    scene, tx, rx = setup_scene(scene_path)
    print()
    
    # Create reflector
    reflector, reflector_ctrl = create_reflector(scene, width=2.0, height=2.0)
    print()
    
    # Set initial position near the corner/intersection area
    initial_position = [5.5, 6.5, 1.5]
    reflector_ctrl.set_position(*initial_position)
    print(f"✓ Reflector positioned at: {initial_position}")
    
    # Set initial orientation (0 degrees for all axes)
    set_reflector_orientation(reflector, alpha_deg=0.0, beta_deg=0.0, gamma_deg=0.0)
    print(f"✓ Reflector orientation: (0.0°, 0.0°, 0.0°)")
    print()
    
    # Display current configuration
    print("Current Configuration:")
    print(reflector_ctrl)
    print()
    
    # Preview the scene
    print("Opening scene preview...")
    
    camera = sionna.rt.Camera(
		position=[14, 10, 33],
		look_at=[14, 10.1, 0.0],
	)
    scene.render_to_file(filename="scene_preview.png", camera=camera, resolution=(800, 600))


if __name__ == "__main__":
    main()
