import numpy as np
import mujoco
from mujoco import viewer
import time
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Create a temporary directory for MuJoCo model
if not os.path.exists("tmp"):
    os.makedirs("tmp")

class RobotArm3Joint:
    def __init__(self, link_lengths=None):
        # Default link lengths if none provided
        self.link_lengths = link_lengths if link_lengths is not None else [0.5, 0.5]
        self.model_path = "tmp/arm_3joint.xml"
        
        # Create the MuJoCo model XML
        self._create_model_xml()
        
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize the simulation
        mujoco.mj_resetData(self.model, self.data)
    
    def _create_model_xml(self):
        """Create the XML model for the robot arm with 3 joints"""
        l1, l2 = self.link_lengths
        
        xml = f"""
        <mujoco model="arm_3joint">
            <option gravity="0 0 -9.81" integrator="RK4"/>
            
            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
                <texture name="plane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512" mark="edge" markrgb="0.8 0.8 0.8"/>
                <material name="plane" texture="plane" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
            </asset>
            
            <worldbody>
                <light diffuse=".8 .8 .8" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.1" material="plane"/>
                
                <body name="base" pos="0 0 0.1">
                    <geom type="cylinder" size="0.1 0.05" rgba="0.5 0.5 0.5 1"/>
                    
                    <body name="link1" pos="0 0 0.05">
                        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0"/>
                        <geom type="capsule" size="0.05" fromto="0 0 0 {l1} 0 0" rgba="0.8 0.3 0.3 1"/>
                        
                        <body name="middle_joint" pos="{l1} 0 0">
                            <joint name="joint_middle" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="0 1.5708"/>
                            <geom type="sphere" size="0.075" rgba="0.8 0.8 0.3 1"/>
                            
                            <body name="link2" pos="0 0 0">
                                <joint name="joint2" type="hinge" axis="0 0 1" pos="0 0 0"/>
                                <geom type="capsule" size="0.05" fromto="0 0 0 {l2} 0 0" rgba="0.3 0.8 0.3 1"/>
                                
                                <site name="end_effector" pos="{l2} 0 0" size="0.08" rgba="0 0 1 0.9"/>
                            </body>
                        </body>
                    </body>
                </body>
                
                <site name="target" pos="1.2 0.6 0.1" size="0.08" rgba="1 0 0 0.9"/>
            </worldbody>
            
            <actuator>
                <motor joint="joint1" name="motor1" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
                <motor joint="joint_middle" name="motor_middle" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
                <motor joint="joint2" name="motor2" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
            </actuator>
        </mujoco>
        """
        
        with open(self.model_path, "w") as f:
            f.write(xml)
    
    def set_joint_angles(self, angles):
        """Set the joint angles of the robot arm"""
        if len(angles) == 3:
            self.data.qpos[0] = angles[0]  # Base joint (rotation around z)
            self.data.qpos[1] = angles[1]  # Middle joint (rotation around y) - constrained to 0-90 degrees
            self.data.qpos[2] = angles[2]  # End joint (rotation around z)
            mujoco.mj_forward(self.model, self.data)
        else:
            raise ValueError("Expected 3 joint angles")
    
    def get_joint_angles(self):
        """Get the current joint angles"""
        return self.data.qpos[:3].copy()
    
    def get_end_effector_pos(self):
        """Get the position of the end effector"""
        return self.data.site_xpos[self.model.site("end_effector").id][:3]
    
    def get_target_pos(self):
        """Get the position of the target"""
        return self.data.site_xpos[self.model.site("target").id][:3]
    
    def calculate_distance_to_target(self):
        """Calculate the Euclidean distance from end effector to target"""
        ee_pos = self.get_end_effector_pos()
        target_pos = self.get_target_pos()
        distance = np.linalg.norm(ee_pos - target_pos)
        return distance
    
    def update_link_lengths(self, link_lengths):
        """Update the link lengths and recreate the model"""
        self.link_lengths = link_lengths
        self._create_model_xml()
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
    
    def update_target_position(self, target_pos):
        """Update the target position"""
        if len(target_pos) >= 2:
            self.model.site_pos[self.model.site("target").id][:2] = target_pos[:2]
            if len(target_pos) >= 3:
                self.model.site_pos[self.model.site("target").id][2] = target_pos[2]
            mujoco.mj_forward(self.model, self.data)


def joint_angle_cost_function(angles, robot):
    """Cost function to minimize for finding optimal joint angles"""
    # Ensure the middle joint angle is within 0-90 degrees
    constrained_angles = angles.copy()
    constrained_angles[1] = np.clip(constrained_angles[1], 0, np.pi/2)
    
    # Set the joint angles
    robot.set_joint_angles(constrained_angles)
    
    # Primary objective: distance to target
    distance = robot.calculate_distance_to_target()
    
    # Secondary objective: penalize extreme joint angles
    angle_penalty = 0.05 * (abs(constrained_angles[0]) + abs(constrained_angles[2]))
    
    # Return combined cost
    return distance + angle_penalty


def link_length_cost_function(link_lengths, target_pos, initial_angles=None):
    """Cost function to minimize for finding optimal link lengths"""
    # Create a robot with these link lengths
    robot = RobotArm3Joint(link_lengths)
    robot.update_target_position(target_pos)
    if initial_angles is None:
        initial_angles = [0, 0.3, 0]  # Default initial angles
    # Find the optimal joint angles for these link lengths
    result = minimize(
        joint_angle_cost_function,
        initial_angles,  # Initial joint angles
        args=(robot,),
        method='L-BFGS-B',
        bounds=[(-np.pi, np.pi), (0, np.pi/2), (-np.pi, np.pi)]  # Middle joint constrained to 0-90 degrees
    )
    robot.set_joint_angles(result.x)
    distance = robot.calculate_distance_to_target()
    length_penalty = 0.05 * sum(link_lengths)
    return distance + length_penalty


def optimize_robot_parameters(target_pos, initial_link_lengths):
    """Optimize the robot arm parameters to reach a target position"""
    print(f"\nOptimizing for target position: {target_pos}")
    print(f"Initial link lengths: {initial_link_lengths}")
    
    # Optimize link lengths
    result = minimize(
        link_length_cost_function,
        initial_link_lengths,
        args=(target_pos,),
        method='L-BFGS-B',
        bounds=[(0.2, 1.5), (0.2, 1.5)]  # Reasonable bounds for link lengths
    )
    
    optimal_link_lengths = result.x
    print(f"Optimization completed!")
    print(f"Optimal link lengths: L1={optimal_link_lengths[0]:.4f}, L2={optimal_link_lengths[1]:.4f}")
    
    # Create robot with optimal link lengths
    optimized_robot = RobotArm3Joint(optimal_link_lengths)
    optimized_robot.update_target_position(target_pos)
    
    # Find optimal joint angles
    angle_result = minimize(
        joint_angle_cost_function,
        [0, 0.3, 0],  # Initial angles
        args=(optimized_robot,),
        method='L-BFGS-B',
        bounds=[(-np.pi, np.pi), (0, np.pi/2), (-np.pi, np.pi)]
    )
    
    optimal_angles = angle_result.x
    print(f"Optimal joint angles: θ1={np.degrees(optimal_angles[0]):.2f}°, " +
          f"θmiddle={np.degrees(optimal_angles[1]):.2f}°, θ2={np.degrees(optimal_angles[2]):.2f}°")
    
    return optimal_link_lengths, optimal_angles


def visualize_robot_motion(robot, target_pos, joint_angles_target=None, viewer_duration=10.0):
    """Visualize the robot motion in MuJoCo viewer"""
    # Set the target position
    robot.update_target_position(target_pos)
    
    # If no target angles provided, find the best ones
    if joint_angles_target is None:
        result = minimize(
            joint_angle_cost_function,
            [0, 0.3, 0],  # Initial angles
            args=(robot,),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi), (0, np.pi/2), (-np.pi, np.pi)]
        )
        joint_angles_target = result.x
    
    # Set initial joint angles (zero position)
    robot.set_joint_angles([0, 0, 0])
    
    # Launch the MuJoCo viewer
    with viewer.launch_passive(robot.model, robot.data) as v:
        # Give the viewer a moment to initialize
        time.sleep(0.5)
        
        # Set the camera
        v.cam.distance = 3.0
        v.cam.azimuth = 45.0
        v.cam.elevation = -30.0
        
        # Initialize time
        start_time = time.time()
        
        # Duration of the movement animation
        motion_time = 3.0
        
        print("\nMuJoCo viewer launched. Displaying robot motion...")
        
        # Keep the viewer open
        while v.is_running() and time.time() - start_time < viewer_duration:
            # Calculate elapsed time
            elapsed = time.time() - start_time
            
            if elapsed < motion_time:
                # Linear interpolation from [0,0,0] to the target joint angles
                alpha = elapsed / motion_time
                current_angles = alpha * joint_angles_target
                robot.set_joint_angles(current_angles)
            else:
                # Keep the final position
                robot.set_joint_angles(joint_angles_target)
            
            # Step the simulation
            mujoco.mj_step(robot.model, robot.data)
            
            # Update the viewer
            v.sync()
            
            # Control the simulation speed
            time.sleep(0.01)
        
        # Calculate final distance to target
        final_distance = robot.calculate_distance_to_target()
        print(f"Final distance to target: {final_distance:.6f}")
        
        return joint_angles_target, final_distance


def visualize_comparison(initial_params, optimized_params, target_pos):
    """Create a visual comparison between initial and optimized robot configurations"""
    initial_link_lengths, initial_angles = initial_params
    optimal_link_lengths, optimal_angles = optimized_params
    
    # Create robots with the respective parameters
    initial_robot = RobotArm3Joint(initial_link_lengths)
    initial_robot.set_joint_angles(initial_angles)
    initial_robot.update_target_position(target_pos)
    
    optimized_robot = RobotArm3Joint(optimal_link_lengths)
    optimized_robot.set_joint_angles(optimal_angles)
    optimized_robot.update_target_position(target_pos)
    
    # Get end effector positions
    initial_ee_pos = initial_robot.get_end_effector_pos()
    optimized_ee_pos = optimized_robot.get_end_effector_pos()
    
    # Calculate distances to target
    initial_distance = initial_robot.calculate_distance_to_target()
    optimized_distance = optimized_robot.calculate_distance_to_target()
    
    # Create visualization
    fig = plt.figure(figsize=(15, 8))
    
    # 3D plot for comparison
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot target
    ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='red', s=100, label='Target')
    
    # Get joint positions for initial robot
    initial_base_pos = np.array([0, 0, 0.1])
    mujoco.mj_forward(initial_robot.model, initial_robot.data)
    initial_mid_pos = initial_robot.data.body('middle_joint').xpos.copy()
    
    # Plot initial robot
    ax.plot([initial_base_pos[0], initial_mid_pos[0]], 
            [initial_base_pos[1], initial_mid_pos[1]], 
            [initial_base_pos[2], initial_mid_pos[2]], 'b-', linewidth=2, label='Initial Robot Link 1')
    ax.plot([initial_mid_pos[0], initial_ee_pos[0]], 
            [initial_mid_pos[1], initial_ee_pos[1]], 
            [initial_mid_pos[2], initial_ee_pos[2]], 'b--', linewidth=2)
    ax.scatter([initial_ee_pos[0]], [initial_ee_pos[1]], [initial_ee_pos[2]], color='blue', s=80, label='Initial End Effector')
    
    # Get joint positions for optimized robot
    optimized_base_pos = np.array([0, 0, 0.1])
    mujoco.mj_forward(optimized_robot.model, optimized_robot.data)
    optimized_mid_pos = optimized_robot.data.body('middle_joint').xpos.copy()
    
    # Plot optimized robot
    ax.plot([optimized_base_pos[0], optimized_mid_pos[0]], 
            [optimized_base_pos[1], optimized_mid_pos[1]], 
            [optimized_base_pos[2], optimized_mid_pos[2]], 'g-', linewidth=2, label='Optimized Robot Link 1')
    ax.plot([optimized_mid_pos[0], optimized_ee_pos[0]], 
            [optimized_mid_pos[1], optimized_ee_pos[1]], 
            [optimized_mid_pos[2], optimized_ee_pos[2]], 'g--', linewidth=2)
    ax.plot(optimized_ee_pos[0], optimized_ee_pos[1], optimized_ee_pos[2],  label='Optimized End Effector')
    
    # Draw lines from end effectors to target
    ax.plot([initial_ee_pos[0], target_pos[0]], [initial_ee_pos[1], target_pos[1]], [initial_ee_pos[2], target_pos[2]], 'b:', alpha=0.5)
    ax.plot([optimized_ee_pos[0], target_pos[0]], [optimized_ee_pos[1], target_pos[1]], [optimized_ee_pos[2], target_pos[2]], 'g:', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot Arm Optimization Comparison')
    
    # Add legend
    ax.legend()
    
    # Add text annotations
    textstr = f"""Initial Configuration:
    - Link Lengths: L1={initial_link_lengths[0]:.2f}, L2={initial_link_lengths[1]:.2f}
    - Joint Angles: θ1={np.degrees(initial_angles[0]):.1f}°, θM={np.degrees(initial_angles[1]):.1f}°, θ2={np.degrees(initial_angles[2]):.1f}°
    - Distance to Target: {initial_distance:.4f}
    
    Optimized Configuration:
    - Link Lengths: L1={optimal_link_lengths[0]:.2f}, L2={optimal_link_lengths[1]:.2f} 
    - Joint Angles: θ1={np.degrees(optimal_angles[0]):.1f}°, θM={np.degrees(optimal_angles[1]):.1f}°, θ2={np.degrees(optimal_angles[2]):.1f}°
    - Distance to Target: {optimized_distance:.4f}
    
    Improvement: {initial_distance - optimized_distance:.4f} units closer"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text2D(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=9,
              verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.show()


def main():
    # Define a challenging target position
    target_pos = [1.2, 1.0, 0.3]  # Target position (x, y, z)
    
    # Initial link lengths
    initial_link_lengths = [0.5, 0.5]
    
    print("="*50)
    print("STEP 1: Testing Initial Robot Configuration")
    print("="*50)
    
    # Create the initial robot
    initial_robot = RobotArm3Joint(initial_link_lengths)
    
    # Visualize the initial robot attempting to reach the target
    initial_angles, initial_distance = visualize_robot_motion(initial_robot, target_pos, viewer_duration=5.0)
    
    print("\n"+"="*50)
    print("STEP 2: Optimizing Robot Parameters")
    print("="*50)
    
    # Optimize the robot parameters
    optimal_link_lengths, optimal_angles = optimize_robot_parameters(target_pos, initial_link_lengths)
    
    print("\n"+"="*50)
    print("STEP 3: Testing Optimized Robot Configuration")
    print("="*50)
    
    # Create the optimized robot
    optimized_robot = RobotArm3Joint(optimal_link_lengths)
    
    # Visualize the optimized robot
    _, optimized_distance = visualize_robot_motion(optimized_robot, target_pos, optimal_angles, viewer_duration=5.0)
    
    # Display comparison
    print("\n"+"="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"Target position: ({target_pos[0]}, {target_pos[1]}, {target_pos[2]})")
    print(f"Initial link lengths: L1={initial_link_lengths[0]:.4f}, L2={initial_link_lengths[1]:.4f}")
    print(f"Initial joint angles: θ1={np.degrees(initial_angles[0]):.2f}°, " +
          f"θmiddle={np.degrees(initial_angles[1]):.2f}°, θ2={np.degrees(initial_angles[2]):.2f}°")
    print(f"Initial distance to target: {initial_distance:.6f}")
    print(f"Optimized link lengths: L1={optimal_link_lengths[0]:.4f}, L2={optimal_link_lengths[1]:.4f}")
    print(f"Optimized joint angles: θ1={np.degrees(optimal_angles[0]):.2f}°, " +
          f"θmiddle={np.degrees(optimal_angles[1]):.2f}°, θ2={np.degrees(optimal_angles[2]):.2f}°")
    print(f"Optimized distance to target: {optimized_distance:.6f}")
    print(f"Improvement: {initial_distance - optimized_distance:.6f} units closer")
    
    # Visualize the comparison
    visualize_comparison((initial_link_lengths, initial_angles), 
                         (optimal_link_lengths, optimal_angles),
                         target_pos)


if __name__ == "__main__":
    main()