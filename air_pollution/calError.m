function [FPN] = calError(trueMat, predictedMat)

subtr = trueMat - predictedMat;
FP = length(find(subtr == -1));
FN = length(find(subtr == 1));
FPN = FP + FN;