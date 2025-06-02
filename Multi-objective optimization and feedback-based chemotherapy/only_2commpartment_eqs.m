function [poli, quo, nor]=only_2commpartment_eqs(Dose)

a = 0.5;
b = 0.477;
c = 0.218;
d = 0.05;
e = 0.0173;
alpha = 0.1;
bita = 1e9;   
gama = 0.27;
miu = 0.4;
lambda = 0.0084;
K = 1e9;


%%

P = zeros(50,50);
Q = zeros(50,50);
M = zeros(50,50);
G = zeros(50,50);
D = zeros(50,50);
T = zeros(50,50);

%% 

P(1,1) = 2e11;
Q(1,1) = 8e11;
G(1,1) = 0.0084*20;
M(1,1) = 1e9;
D(1,1) = 0;
T(1,1) = 0;

%%

List = [2, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85];
% Dose = pattern;
% Dose = [23,20.57,22.28,15,20.52,23,15.41,16.67,16.23,15.33,20.61,18.04,23];
dose_counter = 1;
for i = 2:1:87
    if(find(List == i))
        
        ut = Dose(dose_counter);
        % -----------------------
%         if ut>0
%             tem_M(1,1) = M(1, i-1);
%             tem_D(1,1) = D(1, i-1);
%             tem_G(1,1) = 0; 
%             UT = ut;
%             for j=2:7
%                 tem_D(1, j) = tem_D(1, j-1) + UT - gama*tem_D(1, j-1);
%                 tem_G(1, j) = lambda* tem_D(1,j);
%                 tem_M(1, j) = tem_M(1, j-1) + alpha * tem_M(1, j-1) * (1- tem_M(1, j-1)/K) - (tem_G(1, j)*tem_M(1, j-1));
%                 UT = 0;
%             end
% % %             fprintf("%d) After 7 day Normal cell will be %.3e\n",i,tem_M(1, 7));
%             if log10(tem_M(1, 7))<=8.26
%                 ut = ut - ut*(35/100);
% %                 fprintf("Dose should decrease at: %.2f .\n",ut);
%             end
%         end
        
        
        % Dose(dose_counter) = ut;
        D(1, i) = D(1, i-1) + ut - gama*D(1, i-1);   % define ut
        dose_counter = dose_counter +1;
    else
        ut = 0;
        D(1, i) = D(1, i-1) + ut - gama*D(1, i-1);
    end

   T(1, i) = T(1, i-1) + D(1, i) - miu*T(1, i-1);
   G(1, i) = lambda* D(1,i);

   P(1, i) = P(1, i-1) + (a-b-c)*P(1, i-1) + d*Q(1, i-1) -  G(1, i)*P(1, i-1);   % G(i) will be used
   Q(1, i) = Q(1, i-1) + c*P(1, i) - (d+e)*Q(1, i-1);
   M(1, i) = M(1, i-1) + alpha*M(1, i-1) * (1- M(1, i-1)/bita) - G(1, i)*M(1, i-1);
end

% disp(Dose);

%%

% Z1 =P(1,87) + Q(1,87);
% Z2 = 0;
% for i = 2:1:87
%     Z2 = Z2 + ( M(1,1) - M(1,i));
% end

poli = P(1,:);
quo = Q(1,:);
nor = M(1,:);
% % objval=[Z1];
% objval=[Z1 Z2];
% objval=[10e20 10e5];








