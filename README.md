# Introduction
These codes are meant to implement MPC controllers applied to the _El Manazigui_ robot from the Medical Robotics and Biosignals Lab.

First of all, the robot model is obtained through its URDF file (elmanizigui.urdf) and the toolbox [urdf2casadi](https://github.com/robotology/urdf2casadi-matlab). Then, the MPC implementation is cast by the ```build_mpc.m``` routine.

The reference trajectory is obtained through real experiments and loaded through the data files ```RepetitionData``` and ```RotationData```.

The following explains the code ```el_manizigui_MPC_data_build```:

## Defining the model

```matlab
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
robotacceleration = urdf2casadi.Dynamics.symbolicForwardDynamics(robot_path,0);
```

The code above defines the corresponding states for the robot ($i=1,\dots,4)$: ```q_i``` are the angular positions, while ```qd_i``` are the angular velocities. The control inputs, the torques at each joint, are denominated $\tau_i$. The acceleration, $\ddot{q}$ is obtained through the function ``` urdf2casadi.Dynamics.symbolicForwardDynamics``` and then passed to ```robotacceleration```. The idea is then to build a model as an integrator:

$$
\begin{equation}
\begin{pmatrix}
\dot{q} \\ 
\ddot{q}
\end{pmatrix} = \begin{bmatrix}
0 & 1 \\ 
0 & 0
\end{bmatrix}\begin{pmatrix}
q\\ 
\dot{q}
\end{pmatrix} +
 \begin{bmatrix}
0 \\ 
I
 \end{bmatrix}\tau + \begin{pmatrix}
0 \\
\text{robotacceleration}
 \end{pmatrix}
\end{equation}
$$

The MPC is then built as follows. We define a prediction horizon $N$ through ```opt.N```, the sampling time for integration through ```opt.dt```, the number of states and controls, respectively, through ```opt.n_states``` and ```opt.n_controls```. Then, the model (as described above), the state and control vectors are passed to ```opt.model.function```, ```opt.model.states```, and ```opt.model.controls```. The integration scheme is first-order Euler.

We need only extra parameter to the optimization problem: the reference for $q_i$. This parameter is called ```Ref``` and is declared in ```opt.parameters.name```, accompanied by its dimension (the same size of the state vector) in ```opt.parameters.dim````. 

The stage cost function, $\ell = \sum_{i=1}^N \Vert x_i-ref \Vert_Q + \Vert u \Vert_R$ is declared through ```opt.costs.stage.function```. Note that one must declare the use of the parameter _ref_ in ```opt.costs.stage.parameters```. 

The (upper and lower bounds, variable-wise) constraints are declared through ```opt.constraints.states``` and ```opt.constraints.control```. There is a terminal constraint, of the type $x(N) = ref$, is imposed as a general constraint in ```opt.constraints.general.function```. For this, one must declare the use of _ref_ in ```opt.constraints.general.parameters```, declare it as an _equality_ in ```opt.constraints.general.type``` and declare that it applies only to the last element ($$x(N)$$) through ```opt.constraints.general.elements```.

Finally, since it is an input to the MPC, one must declare _ref_ as an *input vector* through ```opt.input.vector```. 

```matlab
%%
opt. N = 50;  
opt.dt = 0.1;
opt.n_controls  = 4;
opt.n_states    = 8;
opt.model.function = [[qd1;qd2;qd3;qd4]; robotacceleration([q1;q2;q3;q4],[qd1;qd2;qd3;qd4],[0 0 -10],[torque1;torque2;torque3;torque4])];
opt.model.states   =  [q1;q2;q3;q4;qd1;qd2;qd3;qd4];
opt.model.controls = [torque1;torque2;torque3;torque4];
opt.continuous_model.integration = 'euler';

% Define parameters
opt.parameters.name = {'Ref','nul'};
opt.parameters.dim = [opt.n_states, 1; 1 1];

% Define costs
Q = blkdiag(1e10*eye(opt.n_states/2),eye(opt.n_states/2));
R = 0.001*eye(opt.n_controls);

opt.costs.stage.parameters = {'Ref'};
opt.costs.stage.function = @(x,u,varargin) (x-varargin{1})'*Q*(x-varargin{1}) + ...
                                           (u)'*R*(u);
                                       
% control and state constraints
xbound = 50;
opt.constraints.states.upper  = xbound*ones(opt.n_states,1);
opt.constraints.states.lower  = -xbound*ones(opt.n_states,1);
opt.constraints.control.upper = 50*ones(4,1);
opt.constraints.control.lower = -50*ones(4,1);

opt.constraints.general.function{1} = @(x,varargin) x(:,end)-varargin{:};
opt.constraints.general.parameters = {'Ref'};
opt.constraints.general.type{1} = 'equality';
opt.constraints.general.elements{1} = 'end';

% Define inputs to optimization
opt.input.vector = {'Ref'};

% Define the solver and generate it
opt.solver = 'ipopt';
[solver,args_mpc] = build_mpc(opt);
```

Now, we use the experimental data (either ```Repetition.mat``` or ```Rotation.mat```, uncomment one of them) to generate the reference trajectory. Each trajectory is sampled at $2kHz$ (period of $0.0005$s), which is too fast. I re-sample this data to $10Hz$ (period of $0.1$s) and discard the initial elements (which are useless). The variable ``target `` gathers all the references for $q$.

```matlab
%% Get realistic references
% load('RepetitionData.mat')

% EFE = EFE(1:0.1/0.0005:end);
% WFE = WFE(1:0.1/0.0005:end);
% WPS = WPS(1:0.1/0.0005:end);
% WRU = WRU(1:0.1/0.0005:end);
% 
% EFE = EFE(500:end);
% WFE = WFE(500:end);
% WPS = WPS(500:end);
% WRU = WRU(500:end);
% 
% qtarget = deg2rad([EFE;WPS;WRU;WFE]);

load('RotationData.mat')
EFE = EFE(1:0.1/0.0005:end);
WFE = WFE(1:0.1/0.0005:end);
WPS = WPS(1:0.1/0.0005:end);
WRU = WRU(1:0.1/0.0005:end);

EFE = EFE(50:end);
WFE = WFE(50:end);
WPS = WPS(50:end);
WRU = WRU(50:end);

qtarget = deg2rad([EFE;WPS;WRU;WFE]);
```

Finally, the simulation loop is as follows. The variable ```xsimu``` collects the evolution of the states of the robot, ```args_mpc.x0``` gathers the initial conditions for solving the MPC optimization problem, ```args_mpc.p``` gathers all inputs *to the optimization problem* which, in this case, are the state measuments (```xsimu(:,t)```) and the instantaneous reference for $q$ and zero for $\dot{q}$ (```vertcat(qtarget(:,t),zeros(4,1))```).

```matlab
%% Simulation loop
clear i j k
tmax = length(EFE);
xsimu(:,1) = vertcat(qtarget(:,1),zeros(4,1));            % xsimu contains the history of states
u0 = zeros(1,opt.n_controls*opt.N);            % two control inputs for each robot
X0 = zeros(opt.n_states*(opt.N+1),1);     % initialization of the states decision variables

% Start MPC
u = [];
args_mpc.x0 = [X0;u0']; 
    
for t = 1:tmax

    % set the values of the parameters vector
    args_mpc.p = [xsimu(:,t);vertcat(qtarget(:,t),zeros(4,1))];                                              
    target_q(:,t) = qtarget(:,t);
    
    % solve optimization problem
    tic
    sol = solver('x0', args_mpc.x0, 'lbx', args_mpc.lbx, 'ubx', args_mpc.ubx,'lbg', args_mpc.lbg, 'ubg', args_mpc.ubg,'p',args_mpc.p);
    tsol(t) = toc;
    
    % get control sequence from MPC
    aux = full(sol.x(opt.n_states*(opt.N)+opt.n_states+1:opt.n_states*(opt.N+1)+opt.N*opt.n_controls))';
    u(:,t) = aux(:,1:opt.n_controls)';
    aux2 = robotacceleration(xsimu([1:4],t),xsimu(5:8,t),[0 0 -10],[u(:,t)]);
    xsimu(:,t+1) = xsimu(:,t) + opt.dt*[xsimu([5:8],t); aux2.full()];

    args_mpc.x0 = full(sol.x);
    t
end
```
