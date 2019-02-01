function V = CS_neuron(Iapp,t)

% This model is the Connors-Stevens model, similar to Hodgkin-Huxley, but
% more like neurons in the cortex, being type-I. 
% See Dayan and Abbott pp 166-172 then pp.196-198 and p.224.

tmax = length(Iapp);
dt = mean(diff(t));
%% Init params
V_L = -0.070;   % leak reversal potential
E_Na = 0.055;   % reversal for sodium channels
E_K = -0.072;   % reversal for potassium channels
E_A = -0.075;   % reversal for A-type potassium channels

g_L = 3e-6;     % specific leak conductance
g_Na = 1.2e-3;  % specific sodium conductance
g_K = 2e-4;     % specific potassium conductance
g_A = 4.77e-4;  % specific A-tpe potassium conductance
%g_A = 0.0;     % if g_A is zero it switches off the A-current

cm = 10e-9;     % specific membrane capacitance

%% 

V=zeros(size(t)); % voltage vector


V(1) = V_L;    % set the inititial value of voltage     

n=zeros(size(t));   % n: potassium activation gating variable
n(1) = 0.0;         % start off at zero
m=zeros(size(t));   % m: sodium activation gating variable
m(1) = 0.0;         % start off at zero
h=zeros(size(t));   % h: sodim inactivation gating variable
h(1) = 0.0;         % start off at zero

a=zeros(size(t));   % A-current activation gating variable
a(1) = 0.0;         % start off at zero
b=zeros(size(t));   % A-current inactivation gating variable
b(1) = 0.0;         % start off at zero


Itot=zeros(size(t)); % in case we want to plot and look at the total current

for i = 2:length(t); % now see how things change through time
    I_L = g_L*(V_L-V(i-1));
    
    Vm = V(i-1)*1000; % converts voltages to mV as needed in the equations on p.224 of Dayan/Abbott
    
    % Sodium and potassium gating variables are defined by the
    % voltage-dependent transition rates between states, labeled alpha and
    % beta. Written out from Dayan/Abbott, units are 1/ms.
    
    
    if ( Vm == -29.7 ) 
        alpha_m = 0.38/0.1;
    else 
        alpha_m = 0.38*(Vm+29.7)/(1-exp(-0.1*(Vm+29.7)));
    end
    beta_m = 15.2*exp(-0.0556*(Vm+54.7));

    alpha_h = 0.266*exp(-0.05*(Vm+48));
    beta_h = 3.8/(1+exp(-0.1*(Vm+18)));
    
    if ( Vm == -45.7 ) 
       alpha_n = 0.02/0.1;
    else
        alpha_n = 0.02*(Vm+45.7)/(1-exp(-0.1*(Vm+45.7)));
    end
    beta_n = 0.25*exp(-0.0125*(Vm+55.7));
     
    % From the alpha and beta for each gating variable we find the steady
    % state values (_inf) and the time constants (tau_) for each m,h and n.
    
    tau_m = 1e-3/(alpha_m+beta_m);      % time constant converted from ms to sec
    m_inf = alpha_m/(alpha_m+beta_m);
    
    tau_h = 1e-3/(alpha_h+beta_h);      % time constant converted from ms to sec
    h_inf = alpha_h/(alpha_h+beta_h);
    
    tau_n = 1e-3/(alpha_n+beta_n);      % time constant converted from ms to sec
    n_inf = alpha_n/(alpha_n+beta_n);   
    
    m(i) = m(i-1) + (m_inf-m(i-1))*dt/tau_m;    % Update m
    
    h(i) = h(i-1) + (h_inf-h(i-1))*dt/tau_h;    % Update h
    
    n(i) = n(i-1) + (n_inf-n(i-1))*dt/tau_n;    % Update n
    
    % For the A-type current gating variables, instead of using alpha and
    % beta, we just use the steady-state values a_inf and b_inf along with 
    % the time constants tau_a and tau_b that are found empirically
    % (Dayan-Abbott, p. 224)
    
    a_inf = (0.0761*exp(0.0314*(Vm+94.22))/(1+exp(0.0346*(Vm+1.17))))^(1/3.0);
    tau_a = 0.3632*1e-3 + 1.158e-3/(1+exp(0.0497*(Vm+55.96)));
    
    b_inf = (1/(1+exp(0.0688*(Vm+53.3))))^4;
    tau_b = 1.24e-3 + 2.678e-3/(1+exp(0.0624*(Vm+50)));
    
    a(i) = a(i-1) + (a_inf-a(i-1))*dt/tau_a;    % Update a
    b(i) = b(i-1) + (b_inf-b(i-1))*dt/tau_b;    % Update b
    
    I_Na = g_Na*m(i)*m(i)*m(i)*h(i)*(E_Na-V(i-1)); % total sodium current
    
    I_K = g_K*n(i)*n(i)*n(i)*n(i)*(E_K-V(i-1)); % total potassium current
    
    I_A = g_A*a(i)*a(i)*a(i)*b(i)*(E_A-V(i-1)); % total A-type current
    
    Itot(i-1) = I_L+I_Na+I_K+I_A+Iapp(i-1); % total current is sum of leak + active channels + applied current
    
    V(i) = V(i-1) + Itot(i-1)*dt/cm;        % Update the membrane potential, V.

    
end


    