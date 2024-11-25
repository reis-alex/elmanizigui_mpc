clear all; close all; clc;
addpath(genpath('C:\Users\Alex Reis\Documents\MATLAB\urdf2casadi-matlab-master'))
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

robot_path = 'C:\Users\Alex Reis\Desktop\Exoskeleton control\models\urdf\elmanizigui2.urdf';
robot = importrobot(robot_path);
robot_acceleration = urdf2casadi.Dynamics.symbolicForwardDynamics(robot_path,0);

%%
opt. N = 12;  
opt.dt = 0.1;
opt.n_controls  = 4;
opt.n_states    = 8;
opt.model.function = [[qd1;qd2;qd3;qd4]; robot_acceleration([q1;q2;q3;q4],[qd1;qd2;qd3;qd4],[0 0 -10],[torque1;torque2;torque3;torque4])];
opt.model.states   =  [q1;q2;q3;q4;qd1;qd2;qd3;qd4];
opt.model.controls = [torque1;torque2;torque3;torque4];
opt.continuous_model.integration = 'euler';

% Define parameters
opt.parameters.name = {'Ref','nul'};
opt.parameters.dim = [opt.n_states, 1; 1 1];

% Define costs
Q = 1000000*eye(opt.n_states);
R = 0.1*eye(opt.n_controls);

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
tic
[solver,args_mpc] = build_mpc(opt);
toc


%% Create Inverse Kinematics solver
robot.DataFormat = 'row';
gik = robotics.GeneralizedInverseKinematics('RigidBodyTree', robot, ...
    'ConstraintInputs', {'position','joint'});
posTgt = robotics.PositionTarget('LINK4');
posTgt.PositionTolerance = 0.000001;
jntCon = robotics.JointPositionBounds(robot);

theta = linspace(pi/4,-pi/6,10); 
posTgt.TargetPosition    = [0.408*cos(theta(1)),-0.2231, 0.3820+0.408*sin(theta(1))];
jntCon.Bounds            = [3*pi/4,3*pi/2; -pi pi; -20*pi/180 20*pi/180; 0 0;];
g0 = gik(robot.homeConfiguration,posTgt,jntCon);

%% Simulation loop
clear i j k
tmax = 20/opt.dt;
xsimu(:,1) = vertcat(g0',zeros(4,1));            % xsimu contains the history of states
u0 = zeros(1,opt.n_controls*opt.N);            % two control inputs for each robot
X0 = zeros(opt.n_states*(opt.N+1),1);      % initialization of the states decision variables

% Start MPC
u = [];
args_mpc.x0 = [X0;u0']; 
% target = @(t)[sind(5*t);-pi;pi/2;pi/6;0;0;0;0];

    posTgt.TargetPosition = [0.408*cos(theta(end)),-0.2231, 0.3820+0.408*sin(theta(end))];
    qtarget = gik(g0,posTgt,jntCon);
    
for t = 1:tmax
    % use IK solver to determine the joint targets, change every 2 seconds
%     if t==1
%         posTgt.TargetPosition    = [0.408*cos(theta(2)),-0.2231, 0.3820+0.408*sin(theta(2))];
%         tt = 3;
%          qtarget = gik(g0,posTgt,jntCon);
%     end
%     if  mod((t*opt.dt),2) == 0
%         posTgt.TargetPosition    = [0.408*cos(theta(tt)),-0.2231, 0.3820+0.408*sin(theta(tt))];
%          qtarget = gik(g0,posTgt,jntCon);
%         tt = tt+1;
%     end
    % set the values of the parameters vector
    args_mpc.p = [xsimu(:,t);vertcat(qtarget',zeros(4,1))];                                              
    target_q(:,t) = qtarget';
    
    % solve optimization problem
    tic
    sol = solver('x0', args_mpc.x0, 'lbx', args_mpc.lbx, 'ubx', args_mpc.ubx,'lbg', args_mpc.lbg, 'ubg', args_mpc.ubg,'p',args_mpc.p);
    tsol(t) = toc;
    
    % get control sequence from MPC
    aux = full(sol.x(opt.n_states*(opt.N)+opt.n_states+1:opt.n_states*(opt.N+1)+opt.N*opt.n_controls))';
    u(:,t) = aux(:,1:opt.n_controls)';
    aux2 = robot_acceleration(xsimu([1:4],t),xsimu(5:8,t),[0 0 -10],[u(:,t)]);
    xsimu(:,t+1) = xsimu(:,t) + opt.dt*[xsimu([5:8],t); aux2.full()];

    args_mpc.x0 = full(sol.x);
t
%     pause(0.01)
%     t*opt.dt
end

%% 
close all
for k = 1:length(xsimu)
    show(robot,xsimu(1:4,k)');
axis([-0.2 0.6 -0.6 0.2 -0.2 1])
    view([0 90 90])
    drawnow
end