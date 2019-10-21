
function yy()
%% get the regression function for MOF74_FSOLVE_ONLY
clc
result=[];
% try 2*2*2 points
% for Tfeed = linspace(318,373,10);
%      for PL = linspace (0.01,0.1,10);
%         for PH = linspace(1.2,3,10);
%             for yAfeed = linspace(0.1,0.2,10);
%             XX = wholeprocess (Tfeed, PL, PH,yAfeed);
%             result = [result; XX];
%             end
%         end
%     end 
% end

% 10*10*10 points

for Tfeed = linspace(318,373,10);
     for PL = linspace (0.01,0.1,10);
        for PH = linspace(1.2,3,10);
            for yAfeed = linspace(0.1,0.2,10);
            XX = wholeprocess (Tfeed, PL, PH,yAfeed);
            result = [result; XX];
            end
        end
    end 
end

save('mof74naprodresult.mat', 'result');

% load('mof74naprodresult.mat');
% m = result;
% x = m(:,1);% x = Tfeed
% y = m(:,2); % y = PL
% % z = m(:,3); % z = PH
% xx = m (:,4);% xx = yA
% NAprod= m(:,5);
% NBprod = m(:,6);
% Nfeed = m(:,7);
% NRP= m (:,8);
% Wvacuum = m (:,9);
% 
% model1 = polyfitn([x,y,z,xx], NAprod, 2);
% model2 = polyfitn([x,y,z,xx], NBprod, 2);
% model3 = polyfitn([x,y,z,xx], Nfeed, 2);
% model4 = polyfitn([x,y,z,xx], NRP, 2);
% model5 = polyfitn([x,y,z,xx], Wvacuum, 2);
% 
% NAPROD = polyn2sympoly(model1)
% model1.R2
% NBPROD = polyn2sympoly(model2)
% model2.R2
% NFEED = polyn2sympoly(model3)
% model3.R2
% NRPP = polyn2sympoly(model4)
% model4.R2
% WVACUUM = polyn2sympoly(model5)
% model5.R2

%% plot the figure
% for Tfeed = [318,323,348];
%      for PL = linspace(0.01,0.1,10);
%         for PH = [1.2];
%             for yAfeed = [0.12];
%                 for ebed = [0.375];
%             XX = wholeprocess (Tfeed, PL, PH,yAfeed,ebed);
%             result = [result; XX];
%                 end
%             end
%         end
%     end 
% end
% 
% 
% m = result;
% x1 = m(1:10,1);% x = Tfeed
% x2 = m(11:20,1); 
% x3 = m(21:30,1); 
% y1 = m(1:10,2);% y = PL
% y2 = m(11:20,2); 
% y3 = m(21:30,2);
% z1 = m(1:10,3);% z = PH
% z2 = m(11:20,3); 
% z3 = m(21:30,3);
% xx1 = m(1:10,4); % xx = yA
% xx2 = m(11:20,4); 
% xx3 = m(21:30,4);
% yy1 = m(1:10,5); % yy = ebed
% yy2 = m(11:20,5); 
% yy3 = m(21:30,5);
% pur1 = m(1:10,6); % purity
% pur2 = m(11:20,6); 
% pur3 = m(21:30,6);
% rec1 = m(1:10,7); %  recovery
% rec2 = m(11:20,7); 
% rec3 = m(21:30,7);
% sw1 = m(1:10,8); %  specific work
% sw2 = m(11:20,8); 
% sw3 = m(21:30,8);
% wc1 = m(1:10,9); %  working capacity
% wc2 = m(11:20,9); 
% wc3 = m(21:30,9);
% 
% figure(1)
% title('PH 1.2; yA 0.12;ebed 0.375')
% subplot(1,4,1);
%  plot(y1,pur1,'ro-',y2,pur2,'bo-',y3,pur3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('CO2 purity');
% title('Purity');legend('Mof74 dual-303K','323K','348K')
% subplot(1,4,2);
% plot(y1,rec1,'ro-',y2,rec2,'bo-',y3,rec3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('CO2 Recovery');
% title('Recovery');legend('303K','323K','348K')
% subplot(1,4,3);
% plot(y1,sw1,'ro-',y2,sw2,'bo-',y3,sw3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('Specific work(kW/TPDc)');
% title('Specific work');legend('303K','323K','348K')
% subplot(1,4,4);
% plot(y1,wc1,'ro-',y2,wc2,'bo-',y3,wc3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('Working capacity (mol/kg)');
% title('Working capacity');
% legend('303K','323K','348K')
% figure(1)
% title('PH 1.2; yA 0.12;ebed 0.375')
% subplot(1,4,1);
%  plot(y2,pur2,'bo-',y3,pur3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('CO2 purity');
% title('yA');legend('303K','323K','348K')
% subplot(1,4,2);
% plot(y2,rec2,'bo-',y3,rec3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('CO2 Recovery');
% title('yB');legend('303K','323K','348K')
% subplot(1,4,3);
% plot(y2,sw2,'bo-',y3,sw3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('Specific work(kW/TPDc)');
% title('NARP');legend('303K','323K','348K')
% subplot(1,4,4);
% plot(y2,wc2,'bo-',y3,wc3,'ko-');hold all;xlabel('Desorption Pressure(bar)');%ylabel('Working capacity (mol/kg)');
% title('Nfeed1');
% legend('303K','323K','348K')

% 


end 

function XX = wholeprocess(Tfeed, PL, PH,yAfeed)
% clc
%========================first step for Mg-MOF-74==================================
% initial
%yAfeed = 0.12; 
%Tfeed =50+273; % K
T0 = Tfeed;
%PH = 1.2; % bar
P0 = PH; % bar
% PL = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1];% bar
%PL = [0.01];% bar
%N=length(PL);
%for j=1:N;
mm = 200;
% known variables
% adsorbent isotherm parameters for Mg, CO2
m = 1; %kg, mass of sorbent
m1A = 6.8 ; % gmol/kg
m2A = 9.9; % gmol/kg
boA = 2.44E-6; % bar
Q1A = -42000; % J/mol
doA = 1.39E-5; % bar
Q2A = -24000; % J/mol
% adsorbent isotherm parameters for Mg, N2
m1B = 14 ; % gmol/kg
m2B = 0; % gmol/kg
boB = 4.96E-5; % bar
Q1B = -18000; % J/mol
doB = 0/100; % 1/KPa
Q2B = 0; % J/mol
% initial
R = 8.314; % J/mol/K
Cp = 800; % J/kg/K, heat capacity of absorbent Mg

% here I use the Mg
% calculate the inital value for volume
ebed =0.375;  % void fraction of bed
roupellet = 572.7; % kg / m^3, density of absorbent
epellet = 0.72;  % void fraction of absorbent
etotal = ebed+(1-ebed)*epellet;
roubed = (1-ebed)*roupellet;
V = m * etotal/roubed*1e5;    % L*100

% equations of pressure
 P = P0;yA = yAfeed;
yB = 1- yA; PA = P*yA;PB = P*yB;

% Calculate the absorption mole of A as initial value, nadsA
bA = boA*exp(-Q1A/R/T0);
dA = doA*exp(-Q2A/R/T0);
nAads0 = m1A*bA*PA/(1+bA*PA)+m2A*dA*PA/(1+dA*PA); % gmol/kg
bB = boB*exp(-Q1B/R/T0);dB = doB*exp(-Q2B/R/T0);
nBads0 = m1B*bB*PB/(1+bB*PB)+m2B*dB*PB/(1+dB*PB) ;% gmol/kg

NAtotal0 = nAads0*m + yA*P0*V/R/T0; % mol
NBtotal0 = nBads0*m + yB*P0*V/R/T0;% mol

%========================== blowdown step==================================

dN0=0.1; % mol
x=zeros(101,3);x(1,:) = [dN0,T0,yAfeed];
yA0 = yAfeed;
y(1,:) =[nAads0,nBads0,NAtotal0,NBtotal0,T0,yA0];
QA0=-Q1A;QB0=-Q1B;
z(1,:)=[QA0,QB0];
for i=1:mm
    P(1)= PH;
    P(i+1) = PH -i*(PH-PL)/mm;
    options = optimset('display','off','TolF',1e-4);
    x(i+1,:) = fsolve(@f,[0.1, x(i,2:3)],options,P(i+1),y(i,:),V,z(i,:));

    
    % calculate the Wvacuum
    Patm = 1; % bar
    k = 1.4; % ratio of heat capacity
    ita =0.75;
    if P(i)>Patm
       Wvacuum(i)=0;
    else 
       Wvacuum(i)=x(i+1,1)*R/ita*(k/(k-1))*Tfeed*((Patm/P(i))^((k-1)/k)-1);  % J=w*s
    end 

bA(i+1)=boA*exp(-Q1A/R/x(i+1,2));dA(i+1)= doA*exp(-Q2A/R/x(i+1,2));
bB(i+1)=boB*exp(-Q1B/R/x(i+1,2));dB(i+1)= doB*exp(-Q2B/R/x(i+1,2));

PA(i+1)= P(i+1)*x(i+1,3);
PB(i+1)= P(i+1)*(1-x(i+1,3));
y(i+1,1) = m1A*bA(i+1)*PA(i+1)/(1+bA(i+1)*PA(i+1))+m2A*dA(i+1)*PA(i+1)...
    /(1+dA(i+1)*PA(i+1)); % gmol/kg

        % calculate the total moles for A and B;
        y(i+1,2) = m1B*bB(i+1)*PB(i+1)/(1+bB(i+1)*PB(i+1))+m2B*dB(i+1)*...
            PB(i+1)/(1+dB(i+1)*PB(i+1));  % gmol/kg
        
        y(i+1,3) = y(i+1,1)*m + x(i+1,3)*P(i+1)*V/R/x(i+1,2); % Ntotal
        y(i+1,4) = y(i+1,2)*m + (1-x(i+1,3))*P(i+1)*V/R/x(i+1,2);
        y(i+1,5) = x(i+1,2); % T
        y(i+1,6) = x(i+1,3);
        
        
          % calculate the heat of adsorption, QA0,QB0
        MA(i+1)= m1A*bA(i+1)/(1+bA(i+1)*PA(i+1))^2;
        NA(i+1)= m2A*dA(i+1)/(1+dA(i+1)*PA(i+1))^2;
        z(i+1,1) = -(Q1A*MA(i+1)+Q2A*NA(i+1))/(MA(i+1)+NA(i+1));
        MB(i+1)= m1B*bB(i+1)/(1+bB(i+1)*PB(i+1))^2;
        NB(i+1)= m2B*dB(i+1)/(1+dB(i+1)*PB(i+1))^2;
        z(i+1,2) = -(Q1B*MB(i+1)+Q2B*NB(i+1))/(MB(i+1)+NB(i+1));
        
NApro(i) = x(i+1,1)*x(i,3);
NBpro(i) = x(i+1,1)*(1-x(i,3));

end
x;
y;
xx=x(i+1,:);yy=y(i+1,:);zz=z(i+1,:);
Wvac =sum(Wvacuum);
NAprod = sum(NApro);NBprod = sum(NBpro);
workingcapacity = NAprod;
purity = NAprod/(NAprod+NBprod);
%y1(j)= 1/workingcapacity(j)*1e3; % kg/kmol

%=======================repressurization==================================

% inital value from the blow down step, the i+1 result.
%x(1,:) = [dN0,T0,yA0];
x(1,:)=  [0,x(mm+1,2),x(mm+1,3)];
%y(1,:) =[nAads0,nBads0,NAtotal0,NBtotal0,T0,yA0];
y(1,:) =yy;
%  z(1,:)=[QA0,QB0];
 z(1,:)=zz;
for i=1:mm
    P(i+1) = PL + i*(PH-PL)/mm;
    options = optimset('display','off','TolF',1e-4);
    x(i+1,:) = fsolve(@ff,[0, x(i,2:3)],options,P(i+1),y(i,:),V,z(i,:));
 
bA(i+1)=boA*exp(-Q1A/R/x(i+1,2));dA(i+1)= doA*exp(-Q2A/R/x(i+1,2));
bB(i+1)=boB*exp(-Q1B/R/x(i+1,2));dB(i+1)= doB*exp(-Q2B/R/x(i+1,2));

PA(i+1)= P(i+1)*x(i+1,3);
PB(i+1)= P(i+1)*(1-x(i+1,3));
y(i+1,1) = m1A*bA(i+1)*PA(i+1)/(1+bA(i+1)*PA(i+1))+m2A*dA(i+1)*PA(i+1)...
    /(1+dA(i+1)*PA(i+1)); % gmol/kg

        % calculate the total moles for A and B;
        y(i+1,2) = m1B*bB(i+1)*PB(i+1)/(1+bB(i+1)*PB(i+1))+m2B*dB(i+1)*...
            PB(i+1)/(1+dB(i+1)*PB(i+1));  % gmol/kg
        
        y(i+1,3) = y(i+1,1)*m + x(i+1,3)*P(i+1)*V/R/x(i+1,2); % Ntotal
        y(i+1,4) = y(i+1,2)*m + (1-x(i+1,3))*P(i+1)*V/R/x(i+1,2);
        y(i+1,5) = x(i+1,2); % T
        y(i+1,6) = x(i+1,3);


NARP(i) = x(i+1,1)*yAfeed; NBRP(i) = x(i+1,1)*(1-yAfeed);
    
end
x;
y;
yA0 = x(i+1,3); % for the feed step
yB0=1-yA0;
y(i+1,:);
NARPf = y(i+1,3);NBRPf = y(i+1,4);
% mole of co2 input in the repressuration step
NARPP = sum(NARP);NBRPP = sum(NBRP);NRPP = NARPP+NBRPP;
%=======================feed==============================================

% calculate the Wblower
Pfeed = 1;  % bar
% calculate the Nfeed and Nwaste
Nfeed0 = 1.0;
Nwaste0= 1.0;
x0=[Nfeed0,Nwaste0];
NAinitial0 = NAtotal0;
NBinitial0 = NBtotal0;
options = optimset('display','off','TolF',1e-4);
x= fsolve(@fff,x0,options,yAfeed,yA0,NARPf,NBRPf,NAinitial0,NBinitial0);
Nfeed = x(1);
Nwaste = x(2);
% calculate the recovery
Rec = NAprod/(Nfeed*yAfeed+NARPP);
Wblower=(NRPP+Nfeed)*R/ita*(k/(k-1))*Tfeed*((PH/Pfeed)^((k-1)/k)-1);
Specificwork =(Wvac+Wblower)/NAprod/44/24/3.6;

 %fprintf('\n Mg-MOF-74 \n    Tfeed     PL     PH  Purity    Recovery  Specificwork  Workingcapacity\n');
 %fprintf('%11.3f%9.3f%9.3f%9.3f%9.3f%15.3f%12.3f \n',[Tfeed,PL,PH, purity,Rec,Specificwork,workingcapacity]')
% fprintf('\n Mg-MOF-74 \n    Tfeed       PL    PH    Pur    Rec  Specificwork  Workingcapacity\n');
% fprintf('%11.3f%9.3f%9.3%9.3f%9.3f%15.3f%12.3f \n',[Tfeed, PL(j),PH, purity(j),Rec(j),Specificwork(j),workingcapacity(j)]')
% XX=[Tfeed, PL,PH, yAfeed, ebed,Nfeed1,yB01,NARP11,NBRP11];
XX=[Tfeed, PL,PH, yAfeed,NAprod,NBprod,Nfeed,NRPP,Wvac];
%  save('mof74.mat','Tfeed', 'PL','PH', 'purity','Rec','Specificwork','workingcapacity')
% save('mof74.mat','XX','-append')

% figure(1)
% subplot(1,4,1);
%  plot(PL,purity,'b-',PL,purity,'bo');hold all;xlabel('Desorption Pressure(bar)');ylabel('CO2 purity');
% title('Purity');
% subplot(1,4,2);
% plot(PL,Rec,'b-',PL,Rec,'bo');hold all;xlabel('Desorption Pressure(bar)');ylabel('CO2 Recovery');
% title('Recovery');
% subplot(1,4,3);
% plot(PL,Specificwork,'b-',PL,Specificwork,'bo');hold all;xlabel('Desorption Pressure(bar)');ylabel('Specific work(kW/TPDc)');
% title('Specific work');
% subplot(1,4,4);
% plot(PL,workingcapacity,'b-',PL,workingcapacity,'bo');hold all;xlabel('Desorption Pressure(bar)');ylabel('Working capacity (mol/kg)');
% title('Working capacity');
% 
% figure(2)
% plot(PL,y1,'b-',PL,y1,'bo');hold all;xlabel('Desorption Pressure(bar)');ylabel('1/(Nco2) (kg/kmol)');


end



function F=f(x,P,y,V,z)
dN = x(1);
T = x(2);
yA = x(3);
nAads0 = y(1);
nBads0 = y(2);
NAtotal0 = y(3);
NBtotal0 =y(4);
T0 = y(5);
yA0 = y(6);
QA0 = z(1);
QB0 = z(2);

% known variables
% adsorbent isotherm parameters for Mg, CO2
m = 1; %kg, mass of sorbent
m1A = 6.8 ; % gmol/kg
m2A = 9.9; % gmol/kg
boA = 2.44E-6; % bar
Q1A = -42000; % J/mol
doA = 1.39E-5; % bar
Q2A = -24000; % J/mol
% adsorbent isotherm parameters for Mg, N2
m1B = 14 ; % gmol/kg
m2B = 0; % gmol/kg
boB = 4.96E-5; % bar
Q1B = -18000; % J/mol
doB = 0/100; % 1/KPa
Q2B = 0; % J/mol
% initial
R = 8.314; % J/mol/K
Cp = 800; % J/kg/K, heat capacity of absorbent Mg

% Calculate the absorption mole of A as initial value, nadsA
PA = P*yA;PB=P*(1-yA);
bA = boA*exp(-Q1A/R/T);dA = doA*exp(-Q2A/R/T);
nAads = m1A*bA*PA/(1+bA*PA)+m2A*dA*PA/(1+dA*PA); % gmol/kg
bB = boB*exp(-Q1B/R/T);dB = doB*exp(-Q2B/R/T);                                                                                                                                                                                                                                                                                                                                                       
nBads = m1B*bB*PB/(1+bB*PB)+m2B*dB*PB/(1+dB*PB);  % gmol/kg

% calculate the total moles for A and B;
NAtotal = nAads*m + yA*P*V/R/T;
NBtotal = nBads*m + (1-yA)*P*V/R/T;

% three functions we need to solve
F(1)=(NAtotal0-NAtotal)-dN*yA0;
F(2)=(NBtotal0-NBtotal)-dN*(1-yA0);
F(3)= T0 + QA0/Cp/m*(nAads-nAads0) + QB0/Cp/m*(nBads-nBads0)-T;
end

function F=ff(x,P,y,V,z)
dN = x(1);
T = x(2);
yA = x(3);
nAads0 = y(1);
nBads0 = y(2);
NAtotal0 = y(3);
NBtotal0 =y(4);
T0 = y(5);
yA0 = y(6);
QA0 = z(1);
QB0 = z(2);

% known variables
% adsorbent isotherm parameters for Mg, CO2
m = 1; %kg, mass of sorbent
m1A = 6.8 ; % gmol/kg
m2A = 9.9; % gmol/kg
boA = 2.44E-6; % bar
Q1A = -42000; % J/mol
doA = 1.39E-5; % bar
Q2A = -24000; % J/mol
% adsorbent isotherm parameters for Mg, N2
m1B = 14 ; % gmol/kg
m2B = 0; % gmol/kg
boB = 4.96E-5; % bar
Q1B = -18000; % J/mol
doB = 0/100; % 1/KPa
Q2B = 0; % J/mol
% initial
R = 8.314; % J/mol/K
Cp = 800; % J/kg/K, heat capacity of absorbent Mg

% Calculate the absorption mole of A as initial value, nadsA
PA = P*yA;PB=P*(1-yA);
bA = boA*exp(-Q1A/R/T);dA = doA*exp(-Q2A/R/T);
nAads = m1A*bA*PA/(1+bA*PA)+m2A*dA*PA/(1+dA*PA); % gmol/kg
bB = boB*exp(-Q1B/R/T);dB = doB*exp(-Q2B/R/T);                                                                                                                                                                                                                                                                                                                                                       
nBads = m1B*bB*PB/(1+bB*PB)+m2B*dB*PB/(1+dB*PB);  % gmol/kg

% calculate the total moles for A and B;
NAtotal = nAads*m + yA*P*V/R/T;
NBtotal = nBads*m + (1-yA)*P*V/R/T;

% three functions we need to solve
% dN means input mole of the co2. 
F(1)=(NAtotal-NAtotal0)-dN*0.12;
F(2)=(NBtotal-NBtotal0)-dN*(1-0.12);
F(3)= T0 + QA0/Cp/m*(nAads-nAads0) + QB0/Cp/m*(nBads-nBads0)-T;
end

function F=fff(x,yAfeed,yA0,NARPf,NBRPf,NAinitial0,NBinitial0)
Nfeed =x(1);
Nwaste =x(2);

F(1) = NARPf+Nfeed*yAfeed-Nwaste*yA0-NAinitial0;
F(2) = NBRPf+Nfeed*(1-yAfeed)-Nwaste*(1-yA0)-NBinitial0;
end
