classdef denoise_tv < handle

properties
    F      single
    FT     single
    FTF    single
    rho    single
    lambda single
    Z      single
    U      single
    jacobi_iter int32
    admm_iter   int32
end

methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function obj = denoise_tv(N)
    
        obj.Z = zeros(N,N,N);
        obj.U = zeros(N,N,N);
        
        obj.rho = 0.02;
        obj.lambda = 1.5;
        
        obj.jacobi_iter = 4;
        obj.admm_iter = 5;
        
        obj.F = zeros(3,3,3);
        obj.F(2,2,2) = 3;
        obj.F(3,2,2) = -1;
        obj.F(2,3,2) = -1;
        obj.F(2,2,3) = -1;
        
        obj.FT = zeros(3,3,3);
        obj.FT(2,2,2) = 3;
        obj.FT(1,2,2) = -1;
        obj.FT(2,1,2) = -1;
        obj.FT(2,2,1) = -1;
        
        obj.FTF = zeros(3,3,3);
        obj.FTF([1 3],2,2) = -1;
        obj.FTF(2,[1 3],2) = -1;
        obj.FTF(2,2,[1 3]) = -1;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function X = exec(obj,v_in)
    
        obj.Z(:) = 0;
        obj.U(:) = 0;
        
        v = (v_in - mean(v_in(:)))./std(v_in(:));
        
        X = 0.1*randn(size(v_in));
        
        for j = 1:obj.admm_iter
            
            X = 0.1*randn(size(v_in));
            
            tmp = v + convn( (obj.Z-obj.U), obj.rho*obj.FT, 'same' );
            for i = 1:obj.jacobi_iter
                X = ( tmp - convn(X,obj.FTF,'same') )/(1+6);
            end
            
            tmp = convn(X,obj.F,'same');
            tmp_b = tmp - obj.U;
            obj.Z = sign( tmp_b ).*max( abs( tmp_b )-obj.lambda, 0 );
            obj.U = obj.U + tmp - obj.Z;
            
        end
    end
end

end
