%% Author of code
%  Sajal Debnath
%  Roll   : 16CSE025
%  Session: 2015-2016
%  Computer Science and Engineering, University of Barisal
%  Buffer-based Adaptive Fuzzy Classifier.  
%%
clear all
clc
close all

load example.mat
load example.mat
%% Offline
Input.TrainingData  = training_D;   
Input.TrainingLabel = training_L;  
DistanceType='Cosine';
Mode='OfflineTraining';
[Output0]=BAFC_classifier(Input,Mode,DistanceType);

%% Online
load example.mat
load example.mat
Input=Output0;               
Input.TrainingData = Evolving_D;    
Input.TrainingLabel = Evolving_L;   
Mode='EvolvingTraining';
[Output1]=BAFC_classifier(Input,Mode,DistanceType);


%% Validation 
load example.mat
load example.mat
Input=Output1;
Input.TestingData = Testing_D;
Input.TestingLabel= Testing_L;
Mode='Validation';
[Output2]=BAFC_classifier(Input,Mode,DistanceType);
a = Output2.ConfusionMatrix;
fprintf('Correct prediction : %d\n',trace(a));
fprintf('%f\n', trace(a)/ size(Input.TestingData,1)  );