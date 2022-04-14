clc;clear all; close all;
IDX = 6000;
TR  = 0.83;
TE  = 0.17;
All_IDX = 1:IDX;
Tr_IDX = randperm(IDX,TR*IDX); 
Te_IDX = setdiff(All_IDX,Tr_IDX);
save(['./IDX',num2str(IDX),'.mat'],  'Tr_IDX','Te_IDX');