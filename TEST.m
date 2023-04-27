%% test 0
clc;  clear all;
 load('matlab.mat')
%% Read
% Смещаем на 1, чтобы выход являлся следующим значениям курса.
X = reshape((((US(1:end-1)./100)-.5).*2),1,1,1,[]);
Y = reshape((((US(1+1:end)./100)-.5).*2),1,1,1,[]);
trainX = X;
trainY = Y;
tastY = trainY;
tastX = trainX;
%% set
set.maxepochs = inf;
set.batch_size = 10;
set.lrG = 1e-4;
%% Initialization
encoderLG = layerGraph([
    sequenceInputLayer(1,'Name','Input')
    gruLayer(1,'Name','LSTM_1')
    fullyConnectedLayer(36,'Name','FC_1')
    gruLayer(36,'Name','LSTM_22')
    fullyConnectedLayer(1,'Name','FC_22')
    ]);
Predictor = dlnetwork(encoderLG);
avgG.En=[]; avgGS.En=[];

% Train
numIterations = floor(size(trainX,4)/set.batch_size);
out = false; 
epoch = 1; 
global_iter = 0; 
tic; 
while ~out
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*set.batch_size+1:i*set.batch_size;
        XBatch=gpdl(single(trainX(:,1,1,idx)),'CUUT');
        YBatch=gpdl(single(trainY(:,1,1,idx)),'CUUT');
        [Grad,loss] = dlfeval(@modelGradients,XBatch,YBatch,Predictor);

        [Predictor.Learnables,avgG.En,avgGS.En] = adamupdate(Predictor.Learnables, Grad, avgG.En, avgGS.En, global_iter, set.lrG);
     
        LOSS=plotTrain(loss, i, epoch, global_iter, toc/60);
        error = Test(tastX,tastY,Predictor,i,numIterations);
%         if error > 7
%             epoch = set.maxepochs-1;
%             break
%         end
    end
    epoch = epoch+1;
    if epoch == set.maxepochs
        out = true;
    end
    save('NET.mat','Predictor')
end
%% model Gradients
function [Grad,loss]=modelGradients(X,real,Predictor)
fake = forward(Predictor, X);
loss = mean((0.5*(real-fake).^2), 'all');%0.5*(fake-real).^2;
[Grad]=dlgradient(loss,Predictor.Learnables);
end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = gpuArray(dlarray(x,labels));
end
%% TEST
function LOSS=plotTrain (loss, i, epoch, global_iter, Time)
persistent G_loss
if isempty(G_loss)
    G_loss = [];
end
if length(G_loss)>200
    G_loss=[];
end
if mod(i,2)==0 || i==1
    G_loss(end+1)=gather(extractdata(loss));
    figure(i/i)
    plot(G_loss,'-r')
    ylabel('loss')
    title("Эпоха: "+epoch+"| Итерация: "+i+"| Время: "+Time+"m")
    xlabel(['Глобальная итерация = ',num2str(global_iter)])
end
LOSS=G_loss;
end
function error = Test(tastX,tastY,Predictor,i,numIterations)
YBatch=gpdl(single(tastY),'CUUT');
XBatch=gpdl(single(tastX),'CUUT');

fake_e = forward(Predictor, XBatch);
Gen_e=fix(((fake_e.*.5)+.5).*100);
Real_e=fix(((YBatch.*.5)+.5).*100);
error=(mean(Gen_e==Real_e,'all')*100);


if mod(i,2)==0 || i==1 || i==numIterations
    disp(['Процент совпадений',num2str(error)])
    disp('Сгенерированный и оригинальный');
    disp(num2str([Gen_e(:,1),Real_e(:,1)]))
end
end
% %%  test 1
% %% Set up the Import Options and import the data
% opts = delimitedTextImportOptions("NumVariables", 6, "Encoding", "UTF-8");
% opts.DataLines = [1, Inf];
% opts.Delimiter = ",";
% opts.VariableNames = ["Var1", "VarName2", "Var3", "Var4", "Var5", "Var6"];
% opts.SelectedVariableNames = "VarName2";
% opts.VariableTypes = ["string", "double", "string", "string", "string", "string"];
% opts.ExtraColumnsRule = "ignore";
% opts.EmptyLineRule = "read";
% opts = setvaropts(opts, ["Var1", "Var3", "Var4", "Var5", "Var6"], "WhitespaceRule", "preserve");
% opts = setvaropts(opts, ["Var1", "Var3", "Var4", "Var5", "Var6"], "EmptyFieldRule", "auto");
% opts = setvaropts(opts, "VarName2", "TrimNonNumeric", true);
% opts = setvaropts(opts, "VarName2", "ThousandsSeparator", ",");
% %% Read
% US = readtable("C:\Users\523ur\OneDrive\Desktop\KR\Прошлые данные - USD_RUB.csv", opts);
% US = table2array(US(2:5001,1))/1e4;
% Train.Files = reshape(US(1:4700),1,1,1,size(US(1:4700),1));
% Train.Labels = reshape(US(1+1:4700+1),1,1,1,size(US(1:4700),1));
% % Смещаем на 1, чтобы выход являлся следующим значениям курса.
% Test.Files = reshape(US(4701:4999),1,1,1,size(US(4701:4999),1));
% Test.Labels = reshape(US(4701+1:4999+1),1,1,1,size(US(4701:4999),1));
% clear opts US
% %% Объявление сети
% layers = [
%     imageInputLayer([1 1 1],"Name","imageinput")
%     fullyConnectedLayer(100,"Name","fc_1")
%     fullyConnectedLayer(10,"Name","fc_2")
%     fullyConnectedLayer(1,"Name","fc_3")
%     regressionLayer("Name","regressionoutput")
%     ];
% lgraph = layerGraph(layers);
% 
% %% Обучение и тестирование сети STN
% %Анализ сети
% % analyzeNetwork(lgraph);
% plot(lgraph);
% % Опции для обучения 
% options = trainingOptions('sgdm', ... 
%     'MaxEpochs',4,...
%     'MiniBatchSize',64, ...
%     'InitialLearnRate',0.01, ...
%     'Shuffle', 'never', ...
%     'Plots','training-progress');
% %Обучение
% Predictor = trainNetwork(Train,lgraph,options);
% %Проверка
% YPred = classify(Predictor,Test);
% %Расчёт точности
% accuracy = sum(YPred==Test.Labels)/numel(Test.Labels);
% disp(accuracy)
% %% test2
% %% Import data from text file
% clc;  clear all;
% %% Set up the Import Options and import the data
% opts = delimitedTextImportOptions("NumVariables", 6, "Encoding", "UTF-8");
% opts.DataLines = [1, Inf];
% opts.Delimiter = ",";
% opts.VariableNames = ["Var1", "VarName2", "Var3", "Var4", "Var5", "Var6"];
% opts.SelectedVariableNames = "VarName2";
% opts.VariableTypes = ["string", "double", "string", "string", "string", "string"];
% opts.ExtraColumnsRule = "ignore";
% opts.EmptyLineRule = "read";
% opts = setvaropts(opts, ["Var1", "Var3", "Var4", "Var5", "Var6"], "WhitespaceRule", "preserve");
% opts = setvaropts(opts, ["Var1", "Var3", "Var4", "Var5", "Var6"], "EmptyFieldRule", "auto");
% opts = setvaropts(opts, "VarName2", "TrimNonNumeric", true);
% opts = setvaropts(opts, "VarName2", "ThousandsSeparator", ",");
% %% Read
% US = readtable("C:\Users\523ur\OneDrive\Desktop\KR\Прошлые данные - USD_RUB.csv", opts);
% US = table2array(US(2:5001,1))/1e4; %Делим так чтобы остались только дробные части
% % Смещаем на 1, чтобы выход являлся следующим значениям курса.
% X = reshape(US(1:end-1),1,1,1,[]);
% Y = reshape(US(1+1:end),1,1,1,[]);
% trainX = X;
% trainY = Y;
% tastY = trainY;
% tastX = trainX;
% clear opts US
% %% set
% set.maxepochs = inf;
% set.batch_size = 100;
% set.lrG = 1e-4;
% %% Initialization
% Net = layerGraph([
%     sequenceInputLayer(1,'Name','Input')
%     gruLayer(1,'Name','LSTM_1')
%     fullyConnectedLayer(36,'Name','FC_1')
%     fullyConnectedLayer(360,'Name','FC_2')
%     fullyConnectedLayer(36,'Name','FC_3')
%     gruLayer(36,'Name','LSTM_2')
%     fullyConnectedLayer(1,'Name','FC_4')
%     ]);
% Predictor = dlnetwork(Net);
% avgG.En=[]; avgGS.En=[];
% 
% % Train
% numIterations = floor(size(trainX,4)/set.batch_size);
% out = false;
% epoch = 1;
% global_iter = 0;
% tic;
% load('NET.mat');
% while ~out
%     for i=1:numIterations
%         global_iter = global_iter+1;
%         idx = (i-1)*set.batch_size+1:i*set.batch_size;
%         XBatch=gpdl(single(trainX(:,1,1,idx)),'CUUT');
%         YBatch=gpdl(single(trainY(:,1,1,idx)),'CUUT');
%         [Grad,loss] = dlfeval(@modelGradients,XBatch,YBatch,Predictor);
% 
%         [Predictor.Learnables,avgG.En,avgGS.En] = adamupdate(Predictor.Learnables, Grad, avgG.En, avgGS.En, global_iter, set.lrG);
% 
%         LOSS=plotTrain(loss, i, epoch, global_iter, toc/60);
%         error = Test(tastX,tastY,Predictor,i,numIterations);
%     end
%     epoch = epoch+1;
%     if epoch == set.maxepochs
%         out = true;
%     end
%     save('NET.mat','Predictor')
% end
% %% model Gradients
% function [Grad,loss]=modelGradients(X,real,Predictor)
% fake = forward(Predictor, X);
% loss = mean((0.5*(real-fake).^2), 'all');
% % loss = mean((real-fake), 'all');
% [Grad]=dlgradient(loss,Predictor.Learnables);
% end
% %% gpu dl array wrapper
% function dlx = gpdl(x,labels)
% dlx = gpuArray(dlarray(x,labels));
% end
% %% TEST
% function LOSS=plotTrain (loss, i, epoch, global_iter, Time)
% persistent G_loss
% if isempty(G_loss)
%     G_loss = [];
% end
% if length(G_loss)>200
%     G_loss=[];
% end
% if mod(i,5)==0 || i==1
%     G_loss(end+1)=gather(extractdata(loss));
%     figure(i/i)
%     plot(G_loss,'-r')
%     ylabel('loss')
%     title("Эпоха: "+epoch+"| Итерация: "+i+"| Время: "+Time+"m")
%     xlabel(['Глобальная итерация = ',num2str(global_iter)])
% end
% LOSS=G_loss;
% end
% 
% function error = Test(tastX,tastY,Predictor,i,numIterations)
% YBatch=gpdl(single(tastY),'CUUT');
% XBatch=gpdl(single(tastX),'CUUT');
% 
% fake_e = forward(Predictor, XBatch);
% error=(mean(fake_e-YBatch,'all'));
% 
% 
% if mod(i,5)==0 || i==1 || i==numIterations
%     disp(['Средняя погрешность ',num2str(error)])
%     disp('Сгенерированный и оригинальный курсы');
%     disp(num2str([fake_e(:,1),YBatch(:,1)]))
% end
% end