# Introduction
These codes are meant to implement MPC controllers applied to the _El Manazigui_ robot from the Medical Robotics and Biosignals Lab.

First of all, the robot model is obtained through its URDF file (elmanizigui.urdf) and the toolbox [urdf2casadi](https://github.com/robotology/urdf2casadi-matlab). Then, the MPC implementation is cast by the ```build_mpc.m``` routine.

The reference trajectory is obtained through real experiments and loaded through the data files ```RepetitionData``` and ```RotationData```.

The following explains the code ```el_manizigui_MPC_data_build```:

## Defining the model

```
clear all; close all; clc;
import casadi.*

% define state variables
q1  = SX.sym('q1');
qd1 = SX.sym('qd1');
q2  = SX.sym('q2');
qd2 = SX.sym('qd2');
q3  = SX.sym('q3');
qd3 = SX.sym('qd3');
q4  = SX.sym('q4');
qd4 = SX.sym('qd4');
torque1 = SX.sym('torque1'); 
torque2 = SX.sym('torque2');
torque3 = SX.sym('torque3');
torque4 = SX.sym('torque4');

robot_path = fullfile(pwd, 'elmanizigui2.urdf');
robot = importrobot(robot_path);
robot.DataFormat = 'row';
robot_acceleration = urdf2casadi.Dynamics.symbolicForwardDynamics(robot_path,0);
```matlab
