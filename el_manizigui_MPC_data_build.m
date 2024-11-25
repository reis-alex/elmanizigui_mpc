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

robot.DataFormat = 'row';
%%
opt. N = 20;  
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
tic
[solver,args_mpc] = build_mpc(opt);
toc

%% Get realistic references
load('RepetitionData.mat')

EFE = EFE(1:0.1/0.0005:end);
WFE = WFE(1:0.1/0.0005:end);
WPS = WPS(1:0.1/0.0005:end);
WRU = WRU(1:0.1/0.0005:end);

EFE = EFE(500:end);
WFE = WFE(500:end);
WPS = WPS(500:end);
WRU = WRU(500:end);

qtarget = deg2rad([EFE;WPS;WRU;WFE]);

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
    aux2 = robot_acceleration(xsimu([1:4],t),xsimu(5:8,t),[0 0 -10],[u(:,t)]);
    xsimu(:,t+1) = xsimu(:,t) + opt.dt*[xsimu([5:8],t); aux2.full()];

    args_mpc.x0 = full(sol.x);
    t
end

%% 
close all
for k = 1:length(xsimu)
    show(robot,xsimu(1:4,k)');
    axis auto
    view([0 90 90])
    drawnow
end

figure
subplot(221)
plot(EFE,'-b')
hold on
plot(rad2deg(xsimu(1,:)),'--r')

subplot(222)
plot(WPS,'-b')
hold on
plot(rad2deg(xsimu(2,:)),'--r')


subplot(223)
plot(WRU,'-b')
hold on
plot(rad2deg(xsimu(3,:)),'--r')


subplot(224)
plot(WFE,'-b')
hold on
plot(rad2deg(xsimu(4,:)),'--r')

figure
plot(u(1,:),'b')
hold on
plot(u(2,:),'r')
plot(u(3,:),'g')
plot(u(4,:),'k')