load('wine_data.mat');

% t = targets, p = inputs
t = zeros(size(wine_inputs,1),4); % Δημιουργία πίνακα γεμάτο με 
                                  % zeros, διαστάσεων 1599 Χ 4
p = wine_inputs; 

for i=1:size(t,1) % δηλαδή 1599
    if wine_targets_text(i) >= 0 && wine_targets_text(i) <= 4 
        t(i,:) = [1 0 0 0]; % Χαμηλής ποιότητας
    elseif wine_targets_text(i) == 5
        t(i,:) = [0 1 0 0]; % Κατώτερης μέτριας ποιότητας
    elseif wine_targets_text(i) == 6
        t(i,:) = [0 0 1 0]; % Ανώτερης μέτριας ποιότητας
    else
        t(i,:) = [0 0 0 1]; % Υψηλής ποιότητας
    end
end

t = t'; % Αναστροφή του πίνακα targets
p = p'; % Αναστροφή του πίνακα inputs

%save ('wine_data.mat', 'wine_inputs', 'wine_targets_text');

% Training Function
trainfcn = 'trainlm'; 

% Δημιουργία pattern recognition network
hiddenLayer1Size = 70; % default νευρώνες = 10
hiddenLayer2Size = 70;
net = patternnet([hiddenLayer1Size hiddenLayer2Size], trainfcn);

% regularization και normalization των δεδομένων
net.performParam.regularization = 0.001; % default = 0.1
net.performParam.normalization = 'standard';

% Παράμετροι εκπέδευσης
net.trainParam.goal = 1e-6;   % Όταν φτάσει αυτό το error σταματά
net.trainParam.epochs = 1500; % Όταν φτάσει αυτές τις επαναλήψεις σταματά
net.trainParam.show = 40; % Epochs between displays, default = 25
net.trainParam.lr = 0.01; % Learning rate
net.trainParam.max_fail = 30; % default = 6
                              
net.trainParam.show = 1; % ανά πόσα epochs θα τυπώνει το plot

% Κάνουμε setup τα δεδομένα για training, test και validation
net.divideFcn = 'divideint'; 

net.divideParam.trainRatio = 60/100; % Αναλογία για train
net.divideParam.testRatio = 35/100; % Αναλογία για testing                                  
net.divideParam.valRatio = 05/100; % Αναλογία για validation

net.divideMode = 'sample'; % divide every sample

% Performance Function
net.performFcn = 'msereg'; 

% Αρχικοποίηση νευρωνικού δικτύου
net = init(net); 

% Εκπέδευση δικτύου 
[net,tr] = train(net,p,t);

% Το Network
%view(net)
