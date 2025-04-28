% clear all;
% close all;
% clc;
%
% load('x');load('y');
% x=xt;
% y=yt;
% clear xt yt;
% load('xt');load('yt');
% xt=xte;yt=yte;
% clear xte yte;

function [Acc,Sen,Spe,Auc]=ruantiao(x,y,xt,yt,seed,afa,gama,emxil,eta,lamada,deta,hs)
for i=1:3
    xt2=x{i};
    xte2=xt{i};
    mean_data = mean(xt2);
    std_data = std(xt2);
    xt2 = (xt2 - mean_data) ./ std_data;
    xte2 = (xte2 - mean_data) ./ std_data;
    xb{i}=xt2;
    xtb{i}=xte2;
end
x=xb;xt=xtb;
% x表示训练集 xt表示测试集
% 均为转置后的结果
set=3;

%% 初始化
aa=x{1};
[m1,n1] = size(x{1});
% s=rand(length(aa(:,1)),length(aa(:,1)));
s=rand(m1);

% for i=1:length(aa(:,1))
%     d(i,i)=sum(s(i,:)+s(:,i)')/2;
% end
d = zeros(m1);
for i=1:m1
    d(i,i)=sum(s(i,:)+s(:,i)')/2;
end

ls=d-(s+s')/2;
for i=1:set
%     aa=x{i};
%     Q{i}=rand(length(aa(1,:)),hs);
    Q{i}=rand(n1,hs);
end
for i=1:set
%     aa=x{i};
%     w{i}=rand(length(aa(1,:)),hs);
    w{i}=rand(n1,hs);
end
% h=rand(hs,length(aa(:,1)));
h=rand(hs,m1);


%% 算法迭代
for iii=1:4
    %求wv
    for i=1:set
        [mi,~] = size(x{i});
        wa = gengxinu(w{i});
        XXt=x{i}'*x{i};
        xLxt = x{i}'*ls*x{i};
        cv = afa/2*XXt + gama*xLxt;
        bv = -afa*(XXt*Q{i} + x{i}'*h') - 2*gama*xLxt*Q{i} + 1/(mi-1)*wa*deta;%+1/(length(linshixi(:,1))-1)*wa;
        w{i} = -1/2*(cv\bv);
    end

    %求qv
    for i=1:set
%         for j=1:p
            aa=Q{i};
            diagG = zeros(size(aa,1),1);
            for dj=1:size(aa,1)
                diagG(dj,1) = eta*(1+emxil)*(norm(aa(i,:))+2*emxil)/(emxil+norm(aa(i,:)))^2;
            end
            XXt = x{i}'*x{i};
            xLxt = x{i}'*ls*x{i};
            E=afa/2*XXt+gama*xLxt + eta * diag(diagG);
%             F=-afa*x{i}'*x{i}*w{i}+afa/2*x{i}'*h'-2*gama*x{i}'*ls*x{i}*w{i};
            F= -2 * E * w{i} + afa/2*x{i}'*h';
%             Q{i}=-1/2*pinv(E)*F;
            Q{i} = -1/2*(E\F);
            clear G;
%         end
    end

    %求sita
    si=x{1};

    for i=1:set
        a{i} = double(constructW(x{i}));
    end

    for i=1:set
        qq1=(a{i}*a{i}');r1=diag(2*a{i}*s); dd=size(a{i},1);
        sita1= solveTheta(dd, qq1, r1);
        sita{i}=diag(sita1);

    end

    % 求R

    R=y'*h'*pinv((h*h'+eye(length(h(:,1)))));

    %求h

    p1=zeros(hs,length(si(:,1)));

    for i=1:set
        p1=(w{i}-Q{i})'*x{i}'+p1;
    end
    %  sss=(R*y'+afa*p);
    h=pinv(R'*R+2*eye(length(R(1,:))))*(R'*y'+afa*p1);

    %%%%求S
    dim = size(s,2);
%     di=zeros(length(s(1,:)),length(s(1,:)));
    sumd = zeros(dim);
    for i=1:set
        sumd = sumd+sita{i}'*a{i};
    end
%     zero_e = zeros(length(s(1,:)),length(s(1,:)));
    
    sume=zeros(dim);
    for i1=1:set
        ti=(w{i1}-Q{i1})'*x{i1}';
        eij = zeros(dim);
        for i=1:dim
            for j=1:dim
                eij(i,j)=norm(ti(:,i)-ti(:,j))^2;
            end
        end
        sume = sume + eij;
    end
%     argmin_s V\|s\|^2 - 2 s' * sumdi + gamma/2/lambda * s' * ei
%  argmin_s V \|s\|^2 - 2 s' * (sumdi - gamma/4/lambda * ei)
%  argmin_s \|s - 1/V(sumdi - gamma/4/lambda * ei)\|^2,
%  s.t. s'* one=1, s>=0
    tmp = 1/set * (sumd - gama/lamada/4 * sume);
    s = proj_simplex(tmp);
%     for i=1:dim
%         dei=-(di(:,i)-gama/4/lamada*e(:,i));
%         eta1=(1+sum(dei))/length(s(1,:))/2;
%         s(:,i)=dei-eta1*ones(length(dei),1);
%     end

end

pp1=w{1}-Q{1};pp2=w{2}-Q{2};pp3=w{3}-Q{3};

ps1=pp1'*x{1}';ps2=pp2'*x{2}';ps3=pp3'*x{3}';

for i_view = 1:set
    if i_view == 1
        Hallt =R*(xt{i_view}*(w{i_view}-Q{i_view}))';
    else
        Hallt = Hallt + R*(xt{i_view}*(w{i_view}-Q{i_view}))';
    end
end

for i_view = 1:set
    if i_view == 1
        Hall =R*(x{i_view}*(w{i_view}-Q{i_view}))';
    else
        Hall = Hall + R*(x{i_view}*(w{i_view}-Q{i_view}))';
    end
end


%mdl = fitcknn(Hall', y(:,1), 'NumNeighbors', 5);
% 使用支持向量机进行分类
mdl = fitcsvm(Hall', y(:,1), 'KernelFunction', 'rbf', 'Standardize', true);

% 预测测试集的结果
pred_labels = predict(mdl, Hallt');

% 计算精度


[Acc,Sen,Spe]=accsenspe(yt(:,1),pred_labels);
Auc = calculate_auc(yt(:,1), pred_labels);
end



function [ACC,Sen,Spe]=accsenspe(true_labels,pred_labels)
% 计算真阳性（True Positive，TP）
TP = sum((true_labels == 1) & (pred_labels == 1));

% 计算假阳性（False Positive，FP）
FP = sum((true_labels == 0) & (pred_labels == 1));

% 计算真阴性（True Negative，TN）
TN = sum((true_labels == 0) & (pred_labels == 0));

% 计算假阴性（False Negative，FN）
FN = sum((true_labels == 1) & (pred_labels == 0));

% 计算准确度（Accuracy，ACC）
ACC = (TP + TN) / (TP + FP + TN + FN);


% 计算敏感度（Sensitivity，Sen）
Sen = TP / (TP + FN);


% 计算特异度（Specificity，Spe）
Spe = TN / (TN + FP);

end

function auc = calculate_auc(testLabels,predictedLabels)
% 计算假阳性率（FPR）
fpr = sum((predictedLabels == 0) & (testLabels == 0)) / sum(testLabels == 0);

% 计算真阳性率（TPR）
tpr = sum((predictedLabels == 1) & (testLabels == 1)) / sum(testLabels == 1);

% 计算AUC
auc = (tpr + fpr) / 2;
end


function W = constructW(fea)
    % 计算距离矩阵
    D = pdist2(fea, fea).^2;
    
    % 应用指数核
    W = exp(-D);
    
    % 将对角线元素设为0
    W = W - diag(diag(W));
end


function U_update=gengxinu(U)
% k=length(U(1,:));
k = size(U,2);
for temp=1:k
    U_update(:,temp)=decol(U,temp);
end
end
function U_rest=decol(U,i)
[~,m]=size(U);
U_noi=U;
U_noi(:,i)=[];
for temp0=1:(m-1)
    if U(:,i)'*U_noi(:,temp0)<0
        U_noi(:,temp0)=-U_noi(:,temp0);
    end
end
U_rest=sum(U_noi,2);
end

function Theta_ = solveTheta(d, Q, s)

Theta_ = ones(d, 1) / d;
V = ones(d, 1) / d;
sigma2 = 0;
Sigma1 = zeros(d, 1);
ite = 0;
niu = 1e1;
niumax = 1e7;
rho = 1.1;
Id = ones(d, 1);
objvalue = [0];

while ite < 1000
    ite = ite + 1;
    objvalue(end+1) = Theta_' * Q * Theta_ - Theta_' * s;
    if abs(objvalue(end) - objvalue(end-1)) < 1e-5 && abs(sum(Theta_) - 1) < 0.0001 && ite > 2
        break;
    end
    E = 2 * Q + niu * eye(d) + niu * Id * Id';
    f = niu * V + niu * Id - sigma2 * Id - Sigma1 + s;
    Theta_ = inv(E) * f;
    V = max(Theta_ + (1 / niu) * Sigma1, 0);
    Sigma1 = Sigma1 + niu * (Theta_ - V);
    sigma2 = sigma2 + niu * (Theta_' * Id - 1);
    niu = rho * niu;
end
end
