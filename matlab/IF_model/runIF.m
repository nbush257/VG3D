function [V,yhat] = runIF(I_inj,free_params,const_params)
T = length(I_inj);
I_ind = zeros(size(I_inj));
i0 = zeros(size(I_inj));
i1 = i0;
tau_0 = 5;
tau_1 = 50;

V = ones(size(I_inj))*const_params.Vrest;
THETA = ones(size(I_inj))*const_params.THETA_inf;
yhat = false(size(V));
for t = 2:T
    
    %calculate Iind
    di0 = -i0(t-1)/tau_0;
    di1 = -i1(t-1)/tau_1;
    
    % update spike induced currents
    i0(t) = i0(t-1)+di0;
    i1(t) = i1(t-1)+di1;
    I_ind(t) = i0(t)+i1(t);
    
    % V'(t) = -1/tau[V(t)-V_rest]+(I(t)+I_ind(t))/C)
    dV = (-1./free_params.tau)*(V(t-1)-const_params.Vrest) + (I_inj(t-1)+I_ind(t-1))/const_params.C;
    % THETA'(t) = a[V(t)-V_rest]-b[THETA(t)-THETA_inf]
    dTHETA = free_params.a*(V(t-1)-const_params.Vrest) - const_params.b*(THETA(t-1) - const_params.THETA_inf);
    
    % update voltge
    V(t) = V(t-1)+dV;
    % update Threshold?
    THETA(t) = THETA(t-1) + dTHETA;
    
    % If exceed threshold
    if V(t)>THETA(t)
        i0(t) = i0(t)+free_params.A0;
        i1(t) = i1(t)+free_params.A1;
        V(t) = const_params.Vrest;
        THETA(t) = max([THETA(t),const_params.THETA_inf]);
        yhat(t)=1;
    end
   
    
end
