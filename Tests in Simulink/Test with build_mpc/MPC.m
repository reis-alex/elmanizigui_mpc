classdef MPC < matlab.System & matlab.system.mixin.Propagates
    % untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    properties
        % Public, tunable properties.

    end

    properties (DiscreteState)
    end

    properties (Access = private)
        % Pre-computed constants.
        mpc_solver
        mpc_args
        mpc_opt
    end

    methods (Access = protected)
        function num = getNumInputsImpl(~)
            num = 2;
        end
        function num = getNumOutputsImpl(~)
            num = 1;
        end
        function dt1 = getOutputDataTypeImpl(~)
        	dt1 = 'double';
        end
        function dt1 = getInputDataTypeImpl(~)
        	dt1 = 'double';
        end
        function sz1 = getOutputSizeImpl(~)
        	sz1 = [248,1];
        end
        function sz1 = getInputSizeImpl(~)
        	sz1 = [1,1];
        end
        function cp1 = isInputComplexImpl(~)
        	cp1 = false;
        end
        function cp1 = isOutputComplexImpl(~)
        	cp1 = false;
        end
        function fz1 = isInputFixedSizeImpl(~)
        	fz1 = true;
        end
        function fz1 = isOutputFixedSizeImpl(~)
        	fz1 = true;
        end
        
        
        function setupImpl(obj,~,~)
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
            
            % get robot acceleration, build state-space model
            robot_path = fullfile(pwd, 'elmanizigui2.urdf');
            robotacceleration = urdf2casadi.Dynamics.symbolicForwardDynamics(robot_path,0);
            opt.model.function = [[qd1;qd2;qd3;qd4]; robotacceleration([q1;q2;q3;q4],[qd1;qd2;qd3;qd4],[0 0 -10],[torque1;torque2;torque3;torque4])];
            opt.model.states   =  [q1;q2;q3;q4;qd1;qd2;qd3;qd4];
            opt.model.controls = [torque1;torque2;torque3;torque4];
            
            % Define MPC problem
            opt. N = 20;
            opt.dt = 0.1;
            opt.n_controls  = 4;
            opt.n_states    = 8;
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
            opt.costs.stage.function = @(x,u,varargin)  (x-varargin{:})'*Q*(x-varargin{:});
            
            % control and state constraints
            xbound = 20;
            opt.constraints.states.upper  = xbound*ones(opt.n_states,1);
            opt.constraints.states.lower  = -xbound*ones(opt.n_states,1);
            opt.constraints.control.upper = 1.5*ones(4,1);
            opt.constraints.control.lower = -1.5*ones(4,1);
            
            % Define inputs to optimization
            opt.input.vector = opt.parameters.name;
            
            % Define the solver and generate it
            opt.solver = 'ipopt';
            [solver,args_mpc] = build_mpc(opt);
            obj.mpc_solver = solver;
            obj.mpc_args = args_mpc;
            obj.mpc_opt = opt;
        end
        
        % communication betzeen setImpl and stepImpl are made uniquely through obj
        % Implement tasks that need to be performed are each time step 
        function [full_solution] = stepImpl(obj,mpc_input,init_opt)  
            solver = obj.mpc_solver;
            opt = obj.mpc_opt;
            sol = solver('x0', init_opt, 'lbx', obj.mpc_args.lbx, 'ubx', obj.mpc_args.ubx,...
                        'lbg', obj.mpc_args.lbg, 'ubg', obj.mpc_args.ubg, 'p', mpc_input);
  
%             us = full(sol.x(opt.n_states*(opt.N)+opt.n_states+1:opt.n_states*(opt.N+1)+opt.N*opt.n_controls))'; 
            full_solution = full(sol.x);
        end
    end
end