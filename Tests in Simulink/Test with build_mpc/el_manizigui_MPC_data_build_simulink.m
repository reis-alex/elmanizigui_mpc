clear all; close all; clc;
import casadi.*
addpath(genpath([pwd '\urdf2casadi-matlab-master']));
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
% robot = importrobot(robot_path);
% robot.DataFormat = 'row';
robotacceleration = urdf2casadi.Dynamics.symbolicForwardDynamics(robot_path,0);

%%
opt. N = 20;  
opt.dt = 0.1;
opt.n_controls  = 4;
opt.n_states    = 8;
opt.model.function = [[qd1;qd2;qd3;qd4]; robotacceleration([q1;q2;q3;q4],[qd1;qd2;qd3;qd4],[0 0 -10],[torque1;torque2;torque3;torque4])];
opt.model.states   =  [q1;q2;q3;q4;qd1;qd2;qd3;qd4];
opt.model.controls = [torque1;torque2;torque3;torque4];
opt.continuous_model.integration = 'euler';

% Define parameters
for i = 1:opt.N
    opt.parameters.name{i} = ['Ref' int2str(i)];
end
opt.parameters.dim = repmat([opt.n_states, 1;],20,1);

% Define costs
Q = blkdiag(1e10*eye(opt.n_states/2),0.1*eye(opt.n_states/2));
R = 0.001*eye(opt.n_controls);

opt.costs.stage.parameters = opt.parameters.name;
opt.costs.stage.sort_parameter.fixed = [];
opt.costs.stage.sort_parameter.var = 1:20;
opt.costs.stage.function = @(x,u,varargin)  (x-varargin{:})'*Q*(x-varargin{:});% ...
%                                             + u'*R*u;
                                       
% control and state constraints
xbound = 20;
opt.constraints.states.upper  = xbound*ones(opt.n_states,1);
opt.constraints.states.lower  = -xbound*ones(opt.n_states,1);
opt.constraints.control.upper = 1.5*ones(4,1);
opt.constraints.control.lower = -1.5*ones(4,1);

% % this constraint is only for x(N)
% opt.constraints.general.function{1} = @(x,varargin) x(:,end)-varargin{:};
% opt.constraints.general.parameters = {'Ref'};
% opt.constraints.general.type{1} = 'equality';
% opt.constraints.general.elements{1} = 'end';

% this constraint is for x(N) and x(N-1)

% opt.constraints.general.function{1} = @(x,varargin) [x(:,end-6)-varargin{1}; x(:,end-5)-varargin{2}; ...
%                                                      x(:,end-4)-varargin{3}; x(:,end-3)-varargin{4};
%                                                      x(:,end-2)-varargin{5}; x(:,end-1)-varargin{6}];
% opt.constraints.general.parameters = opt.parameters.name;
% opt.constraints.general.type{1} = 'equality';
% opt.constraints.general.elements{1} = 'all';

% Define inputs to optimization
opt.input.vector = opt.parameters.name;

% Define the solver and generate it
opt.solver = 'ipopt';
[solver,args_mpc] = build_mpc(opt);

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

%% Simulation loop
clear i j k
xsimu(:,1) = vertcat(qtarget(:,1),zeros(4,1));            % xsimu contains the history of states
u0 = zeros(1,opt.n_controls*opt.N);            % two control inputs for each robot
X0 = zeros(opt.n_states*(opt.N+1),1);     % initialization of the states decision variables

% Start MPC
u = [];
init_var = [X0;u0']; 

for t = 1:length(EFE)-20

    % set the values of the parameters vector
    mpc_input       = [xsimu(:,t);reshape(vertcat(qtarget(:,t:t+19),zeros(4,20)),8*20,1)];                                              
    target_q(:,t)   = qtarget(:,t);
       
    % get control sequence from MPC
    sim_mpc = sim('mpc_block');
    aux = mpc_solution.Data(opt.n_states*(opt.N)+opt.n_states+1:opt.n_states*(opt.N+1)+opt.N*opt.n_controls);
    u(:,t) = aux(:,1:opt.n_controls)';
    aux2 = robotacceleration(xsimu([1:4],t),xsimu(5:8,t),[0 0 -10],[u(:,t)]);
    xsimu(:,t+1) = xsimu(:,t) + opt.dt*[xsimu([5:8],t); aux2.full()];

    init_var = mpc_solution.Data(:,:,end);
    t
end

%% 
% close all
% for k = 1:length(xsimu)
%     show(robot,xsimu(1:4,k)');
%     axis auto
%     view([0 90 90])
%     drawnow
% end

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
stairs(u(1,:),'b')
hold on
stairs(u(2,:),'r')
stairs(u(3,:),'g')
stairs(u(4,:),'k')