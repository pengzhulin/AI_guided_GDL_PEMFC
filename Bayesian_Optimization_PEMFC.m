clear all;
rng(10); % create random seeds

% Create normalized variables for  porosity, raidus/10, orientation/90
vars = [optimizableVariable('x1',[0.5,0.95]), ...
        optimizableVariable('x2',[0.1,0.95]), ...
        optimizableVariable('x3',[0,0.95])];

% Bayesian optimization setting
results = bayesopt(@objectiveFunction, vars, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 50);  

% Output bestX and bestObjective
bestX = results.XAtMinObjective;
bestObjective = results.MinObjective;

disp('bestX');
disp(bestX);
disp('bestObjective');
disp(bestObjective);

% using comsol api to conduct multiphysical simulation
function y = objectiveFunction(x)
    %load NN model
    loadedNet = load('folder_path\saved_net.mat');
    NN = loadedNet.net;
    x_NN = predict(NN, [x.x1 x.x2 x.x3]); % get  normalized predicted results from NN
    X_com_1 = x_NN(1); % (1) get normalized porosity predicted by NN
    X_com_2 = 1/x_NN(2); % (1)get normalized tortuosity predicted by NN (2) calculate tortuosity
    X_com_3 = 10^(x_NN(3)*-15);% (1)get normalized kappa_z predicted by NN (2) calculate kappa_z
    X_com_4 = 10^(x_NN(4)*-15);% (1)get normalized kappa_y predicted by NN (2) calculate kappa_y
    X_com_5 = 10^(x_NN(5)*-15);% (1)get normalized kappa_x predicted by NN (2) calculate kappa_x
    %load comsol model
    model = mphopen('folder_path\multiphysical_fuel_cell_comsol_model.mph');
    % set calculated parameters in comsol 
    model.param("par2").set("eps_gdl", num2str(X_com_1));
    model.param("par2").set("f_tort", num2str(X_com_2));
    model.param("par2").set("kappa_px", num2str(X_com_3) + "[m^2]");
    model.param("par2").set("kappa_py", num2str(X_com_4) + "[m^2]");
    model.param("par2").set("kappa_pz", num2str(X_com_5) + "[m^2]");
    model.sol("sol1").runAll(); % run comsol model
    model.result().table("tbl1").clearTableData();
    model.result().numerical("gev1").set("table", "tbl1"); % read results
    model.result().numerical("gev1").setResult();
    single_polar = mphtable(model, 'tbl1').data; % read polraization curves
    limit_cd = max(single_polar(:,3)); % get limiting current density
    y = -limit_cd; % setting objective function to get maxinum limiting current density

end

