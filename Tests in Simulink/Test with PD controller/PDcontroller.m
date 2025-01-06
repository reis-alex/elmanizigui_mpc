classdef PDcontroller < matlab.System & matlab.system.mixin.Propagates
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
        Kp
        Kd
    end

    methods (Access = protected)
        function num = getNumInputsImpl(~)
            num = 2; %set number of inputs to this block, in the example below it is x and ref
        end
        function num = getNumOutputsImpl(~)
            num = 1; %set number of outputs to this block, in the example below it is u
        end
        function dt1 = getOutputDataTypeImpl(~)
        	dt1 = 'double';
        end
        function dt1 = getInputDataTypeImpl(~)
        	dt1 = 'double';
        end
        function sz1 = getOutputSizeImpl(~)
        	sz1 = [1,1];
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
            % Implement tasks that need to be performed only once, 
            % such as pre-computed constants.
            [v1,v2] = getgains();
            obj.Kp = v1;
            obj.Kd = v2;
        end
        
        % Important: communication between setImpl and stepImpl are made uniquely through the variable obj

        function u = stepImpl(obj,x,ref) 
            u = -obj.Kp*(x(1)-ref(1)) - obj.Kd*(x(2)-ref(2));
        end

%         function resetImpl(obj)
%             % Initialize discrete-state properties.
%         end
    end
end
