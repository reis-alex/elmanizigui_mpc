clear all; close all; clc;
addpath(genpath('C:\Users\Alex Reis\Documents\MATLAB\urdf2casadi-matlab-master'))
import casadi.*
N = 12;  

% model right-hand side
dt = 0.01;


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

% robot_path = 'C:\Users\Alex Reis\Desktop\Exoskeleton control\models\urdf\elmanizigui2.urdf';
robot_path = 'C:\Users\Alex Reis\Desktop\Exoskeleton control\models\urdf\elmanizigui2.urdf';
robot = importrobot(robot_path);
robot_acceleration = urdf2casadi.Dynamics.symbolicForwardDynamics(robot_path,0);
% model = [0 1; 0 0]*states + [0;1]*controls; %
model = [[qd1;qd2;qd3;qd4]; robot_acceleration([q1;q2;q3;q4],[qd1;qd2;qd3;qd4],[0 0 -10],[torque1;torque2;torque3;torque4])];

%%
tmax = 5;

%% Create Inverse Kinematics solver
robot.DataFormat = 'row';
gik = robotics.GeneralizedInverseKinematics('RigidBodyTree', robot, ...
    'ConstraintInputs', {'position','joint'});
posTgt = robotics.PositionTarget('LINK4');
posTgt.PositionTolerance = 0.01;
jntCon = robotics.JointPositionBounds(robot);

% use IK solver to determine the joint targets
theta = linspace(pi/4,-pi/6,10); 
jntCon.Bounds            = [3*pi/4,3*pi/2; -pi pi; -20*pi/180 20*pi/180; 0 0;];
for i = 1:numel(theta)
    posTgt.TargetPosition    = [0.408*cos(theta(i)),-0.2231, 0.3820+0.408*sin(theta(i))];
    if i == 1
    g00(:,i) = gik(robot.homeConfiguration,posTgt,jntCon);
    else
    g00(:,i) = gik(g00(:,i-1)',posTgt,jntCon);
    end
end

for i = 1:4
    g0(i,:) = interp1(theta,g00(i,:),linspace(pi/4,-pi/6,tmax/dt));
end
g1 = diff(g0,1,2)/dt;
g2 = diff(g1,1,2)/dt;
%% Simulation loop
clear i j k xsimu
close all
n_controls = 4;
n_states = 8;
xsimu(:,1) = vertcat(robot.homeConfiguration',zeros(4,1));            % xsimu contains the history of states

% Start MPC
K = blkdiag(20,100,1,1);
K2 = blkdiag(10,10,1,1);

xteste(:,1) = xsimu(1,:);

for t = 1:tmax/dt
    [H,C] = urdf2casadi.Dynamics.HandC(robot_path, xsimu(1:4,t), xsimu(5:8,t), [0 0 -10]);
    u(:,t) = C + H*g2(:,t) - H*K*(xsimu(1:4,t) - g0(:,t)) - H*K2*(xsimu(5:8,t)- g1(:,t));
    aux2 = robot_acceleration(xsimu([1:4],t),xsimu(5:8,t),[0 0 -10],[u(:,t)]);
    xsimu(:,t+1) = xsimu(:,t) + dt*[xsimu([5:8],t); aux2.full()];
    show(robot,xsimu(1:4,t)');
%     hold on
%     plot3(posTgt.TargetPosition(1),posTgt.TargetPosition(2),posTgt.TargetPosition(3),'ok')
%     hold off
    axis auto
    view([0 90 90])
    drawnow
    
%     pause(0.5)
    t*dt
end

figure
plot(g0(1,:),'k')
hold on
plot(g0(2,:),'b')
plot(g0(3,:),'r')
plot(g0(4,:),'g')
plot(xsimu(1,:),'k--')
plot(xsimu(2,:),'b--')
plot(xsimu(3,:),'r--')
plot(xsimu(4,:),'--g')


