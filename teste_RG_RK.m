clear all; close all; clc;
addpath(genpath('C:\Users\Alex Reis\Documents\MATLAB\urdf2casadi-matlab-master'))
import casadi.*
N = 3;  

% model right-hand side
dt = 0.005;


% define state variables
q1  = SX.sym('q1');
qd1 = SX.sym('qd1');
q2  = SX.sym('q2');
qd2 = SX.sym('qd2');
q3  = SX.sym('q3');
qd3 = SX.sym('qd3');
q4  = SX.sym('q4');
qd4 = SX.sym('qd4');
ref1 = SX.sym('ref1'); 
ref2 = SX.sym('ref2');
ref3 = SX.sym('ref3');
ref4 = SX.sym('ref4');
ref5 = SX.sym('ref5');
ref6 = SX.sym('ref6');
ref7 = SX.sym('ref7');
ref8 = SX.sym('ref8');
ref9 = SX.sym('ref9');
ref10 = SX.sym('ref10');
ref11 = SX.sym('ref11');
ref12 = SX.sym('ref12');

% robot_path = 'C:\Users\Alex Reis\Desktop\Exoskeleton control\models\urdf\elmanizigui2.urdf';
robot_path = 'C:\Users\Alex Reis\Desktop\Exoskeleton control\models\urdf\elmanizigui2.urdf';
robot = importrobot(robot_path);
robot_acceleration = urdf2casadi.Dynamics.symbolicForwardDynamics(robot_path,0);

K =  blkdiag(1,1,1,1);
K2 = blkdiag(1,1,1,1);

model = [[qd1;qd2;qd3;qd4]; (-K*([q1;q2;q3;q4]-[ref2;ref2;ref3;ref4]) - K2*([qd1;qd2;qd3;qd4]-[ref5;ref6;ref7;ref8]) + [ref9;ref10;ref11;ref12])]; % 

%% 
states = [q1;q2;q3;q4;qd1;qd2;qd3;qd4];
filtered_ref = [ref1;ref2;ref3;ref4;ref5;ref6;ref7;ref8;ref9;ref10;ref11;ref12]; %

n_states = length(states);
n_controls = length(filtered_ref);

% model function f(x,u)
f = Function('f',{states,filtered_ref},{model}); 

U = SX.sym('U',n_controls,1);               % Decision variables (controls)
Param = SX.sym('P',20);               % parameters (which include the initial state and the reference state)
X = SX.sym('X',n_states,(N+1));             % A vector that represents the states over the optimization problem.

obj = 0;        % Objective function
g = [];         % constraints vector

state = X(:,1);                                    % initial state
g     = [g; state-Param(1:n_states)];            % initial condition constraints
R = blkdiag(1e1*eye(4),0.0000001*eye(4),eye(4));
refreal = [Param(9:20)];
for k = 1:N
    state           = X(:,k); 
    filtered         = U(:,1);
    obj             = obj + (filtered-refreal)'*R*(filtered-refreal); % calculate obj
    state_next      = X(:,k+1);
    
%     k1 = f(state,filtered);
%     k2 = f(state+dt/2*k1, filtered);
%     k3 = f(state+dt/2*k2, filtered);
%     k4 = f(state+dt*k3, filtered);
    
    f_value         = f(state,filtered);
    st_next_RK   = state + dt*f_value;%/6*(k1+2*k2+2*k3+k4);
    g               = [g; state_next-st_next_RK];   % compute constraints
end
g = [g; X(1:4,N+1)-filtered(1:4)]; %target(1:n_states)
OPT_variables   = [reshape(X(:,1:end),n_states*(N+1),1);
                    reshape(U,n_controls,1);];

nlp_prob        = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', Param);

% constraints
args = struct;
% equality for x(k+1)-x(k)
args.lbg(1:n_states*(N+1)) = 0;                     % -1e-20 % Equality constraints
args.ubg(1:n_states*(N+1)) = 0;                     % 1e-20 % Equality constraints

%end point constraint
args.lbg(n_states*(N+1)+1:n_states*(N+1)+n_states/2) = 0;
args.ubg(n_states*(N+1)+1:n_states*(N+1)+n_states/2) = 0;

% bounds for variables (states)

vconst = [1; inf; inf; inf];
for k = 1:n_states
    if sum(k == 1:4)
        args.lbx(k:n_states:n_states*(N+1),1) = -inf; 
        args.ubx(k:n_states:n_states*(N+1),1) = inf; 
    end
    if sum(k == 5:8)
        args.lbx(k:n_states:n_states*(N+1),1) = -vconst(k-4)*ones(N+1,1); 
        args.ubx(k:n_states:n_states*(N+1),1) = vconst(k-4)*ones(N+1,1); 
    end
end

% bounds for variables (controls)
lbounds = [-inf*ones(4,1);-inf*ones(4,1);-inf*ones(4,1)];
ubounds = -lbounds;

for k = 1:n_controls
    args.lbx(n_states*(N+1)+k:n_controls:n_states*(N+1)+n_controls*1,1) = lbounds(k);%*ones(N,1);
    args.ubx(n_states*(N+1)+k:n_controls:n_states*(N+1)+n_controls*1,1) = ubounds(k);%*ones(N,1); 
end


% create solver, define options
options.terminationTolerance = 1e-5;
options.boundTolerance = 1e-5;
options.printLevel = 'none';
options.error_on_fail = 0;
solver = qpsol('solver','qpoases',nlp_prob,options);

%% Create Inverse Kinematics solver
robot.DataFormat = 'row';
gik = robotics.GeneralizedInverseKinematics('RigidBodyTree', robot, ...
    'ConstraintInputs', {'position','joint'});
posTgt = robotics.PositionTarget('LINK4');
posTgt.PositionTolerance = 0.01;
jntCon = robotics.JointPositionBounds(robot);

% use IK solver to determine the joint targets
theta = linspace(pi/4,-pi/6,5); 
jntCon.Bounds            = [3*pi/4,3*pi/2; -pi pi; -20*pi/180 20*pi/180; 0 0;];
for i = 1:numel(theta)
    posTgt.TargetPosition    = [0.408*cos(theta(i)),-0.2231, 0.3820+0.408*sin(theta(i))];
    if i == 1
    g00(:,i) = gik(robot.homeConfiguration,posTgt,jntCon);
    else
    g00(:,i) = gik(g00(:,i-1)',posTgt,jntCon);
    end
end

tmax = 2.5;
g0 = g00(:,end).*ones(4,tmax/dt);
g1 = diff(g0,1,2)/dt;
g2 = diff(g1,1,2)/dt;

%% Simulation loop
clear i j k

xsimu(:,1) = vertcat(g00(:,1),g1(:,1));            % xsimu contains the history of states
u0 = zeros(1,n_controls);            % two control inputs for each robot
X0 = zeros(n_states*(N+1),1);      % initialization of the states decision variables

% Start MPC
u = [];
args.x0 = [X0;u0';]; 

for t = 1:(tmax/dt)-2

    % set the values of the parameters vector
    args.p = [xsimu(:,t);vertcat(g0(:,t),g1(:,t),g2(:,1))]; 
    tic
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    tsol(t) = toc;
    
    % get reference sequence from MPC
    aux = full(sol.x(n_states*(N)+n_states+1:n_states*(N+1)+n_controls))';
    ref(:,t) = aux(:,1:n_controls);
    
    xsimu2(:,1) =  xsimu(:,t);
    tt = full(sol.x);
    clear k1 k2 k3 k4 H2 C2 ximu2 u2 aux22 kk k1 k2 k3 k4
    
    % compute inner loop control
    tic
    [H,C] = urdf2casadi.Dynamics.HandC(robot_path, xsimu(1:4,t), xsimu(5:8,t), [0 0 -10]);
    u(:,t) = C + H*(ref(9:12,t) - K*(xsimu(1:4,t) - ref(1:4,t)) - K2*(xsimu(5:8,t) - ref(5:8,t))); %
    thc(t)= toc;
    
    % simulate robot
    aux2 = full(robot_acceleration(xsimu([1:4],t),xsimu(5:8,t),[0 0 -10],[u(:,t)]));

    xsimu(:,t+1) = xsimu(:,t) + dt*[xsimu([5:8],t); aux2];
    show(robot,xsimu(1:4,t)');
    
    % do plots, reinitialize initial condtions for OCP
    args.x0 = full(sol.x);
    axis auto
    view([0 90 90])
    drawnow
    t*dt

end

%%
figure
subplot(121)
hold on
stairs(g0(1,:),'k')
stairs(ref(1,:),'--r')
stairs(xsimu(1,:),'b')
subplot(122)
hold on
stairs(g1(1,:),'k')
stairs(ref(5,:),'--r')
stairs(xsimu(5,:),'b')