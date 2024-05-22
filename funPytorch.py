import os
from pathlib import Path
import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sl
import sklearn.metrics
import pickle
import time
import math
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torchsummary
from tensorboardX import SummaryWriter
from collections import defaultdict

import loaders
import importlib

from ray import tune
from ray.tune import ExperimentAnalysis

class MRELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return torch.mean(torch.abs(y - yhat) / y)

class SLoss(nn.Module):
    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf
        B_lims = [520, 536]
        W1_lims = [0.004, 0.009]
        self.g = 1.0705e-3
        self.omega2 = 2 * np.pi * torch.linspace(self.g * min(B_lims) - 5 * max(W1_lims), self.g * max(B_lims) + 5 * max(W1_lims), 500).to(device)
        if conf.normalizeY:
            self.minMax = conf.multiCollapseMinMax

    def _funcGauss(self, x, y0, a, xc, w):
        return y0 + a * torch.exp(-0.5 * ((x - 2 * np.pi * xc) / (2 * np.pi * w)) ** 2)

    def _funcNoise(self, x, y0, a1, x1, w1):
        return y0 + self._funcGauss(x, 0, a1, x1, w1)

    def forward(self, yhatBatch, yBatch):
        error = 0
        for yhat, y in zip(yhatBatch, yBatch):
            if self.conf.has("fixedB"):
                Y0, Y0_hat = y[0], yhat[0]
                A, A_hat = y[1], yhat[1]
                B = B_hat = self.conf.fixedB
                W1, W1_hat = y[2], yhat[2]
            else:
                Y0, Y0_hat = y[0], yhat[0]
                A, A_hat = y[1], yhat[1]
                B, B_hat = y[2], yhat[2]
                W1, W1_hat = y[3], yhat[3]

            if self.conf.normalizeY:
                Y0 = self.denormalize(Y0, 0)
                Y0_hat = self.denormalize(Y0_hat, 0)
                A = self.denormalize(A, 1)
                A_hat = self.denormalize(A_hat, 1)
                if not self.conf.has("fixedB"):
                    B = self.denormalize(B, 2)
                    B_hat = self.denormalize(B_hat, 2)
                W1 = self.denormalize(W1, 3)
                W1_hat = self.denormalize(W1_hat, 3)

            vl = B_hat * self.g
            para_A = [0.0, A_hat, vl, W1_hat]
            vl = B * self.g
            para_B = [0.0, A, vl, W1]

            error += abs(Y0_hat - Y0) * (8.5 - 0.001) + sum(abs(self._funcNoise(self.omega2, *para_B) - self._funcNoise(self.omega2, *para_A))) * (self.omega2[1] - self.omega2[0])

        return error / len(yBatch)

    def denormalize(self, value, index):
        return value * (self.minMax[index][1] - self.minMax[index][0]) + self.minMax[index][0]

class MyLoss(nn.Module):
    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf
        self.lossFun = self._get_loss_function(conf, device)

    def _get_loss_function(self, conf, device):
        if conf.taskType == "classification":
            return nn.CrossEntropyLoss()
        elif conf.taskType == "prediction":
            return nn.KLDivLoss(reduction='batchmean')
        elif conf.taskType == "regressionL1":
            return nn.L1Loss()
        elif conf.taskType == "regressionL2":
            return nn.MSELoss()
        elif conf.taskType == "regressionMRE":
            return MRELoss()
        elif conf.taskType == "regressionS":
            return SLoss(conf, device)
        else:
            raise ValueError(f"taskType {conf.taskType} not valid")

    def forward(self, yhatBatch, yBatch):
        return self.lossFun(yhatBatch, yBatch)

def filesPath(conf):
    if conf.has("runningPredictions") and conf.runningPredictions and conf.has("useTune") and conf.useTune:
        mode = "max" if conf.bestSign == '>' else "min"
        analysis = Analysis(os.path.join("tuneOutput", conf.path))
        try:
            tunePath = analysis.get_best_logdir(metric=f"valid/{conf.bestKey}", mode=mode)
        except KeyError:
            trialDF = analysis.trial_dataframes
            bestMetric = np.inf if mode == 'min' else -np.inf
            for i in range(len(list(trialDF.keys()))):
                try:
                    currMetric = min(trialDF[list(trialDF.keys())[i]][f"valid/{conf.bestKey}"]) if mode == 'min' else max(trialDF[list(trialDF.keys())[i]][f"valid/{conf.bestKey}"])
                    if (mode == 'min' and currMetric < bestMetric) or (mode == 'max' and currMetric > bestMetric):
                        bestMetric = currMetric
                        tunePath = list(trialDF.keys())[i]
                except KeyError:
                    print(f"Skipped: {list(trialDF.keys())[i]}")
        return os.path.join(tunePath, "files")
    elif conf.has("useTune") and conf.useTune:
        return "files"
    else:
        return os.path.join("files", conf.path)

def makeModel(conf, device):
    modelPackage = importlib.import_module(f"models.{conf.model}")
    if conf.has("runningPredictions") and conf.runningPredictions and conf.has("useTune") and conf.useTune:
        mode = "max" if conf.bestSign == '>' else "min"
        analysis = Analysis(os.path.join("tuneOutput", conf.path))
        try:
            conf = conf.copy(analysis.get_best_config(metric=f"valid/{conf.bestKey}", mode=mode))
        except KeyError:
            trialDF = analysis.trial_dataframes
            bestMetric = np.inf if mode == 'min' else -np.inf
            for i in range(len(list(trialDF.keys()))):
                try:
                    currMetric = min(trialDF[list(trialDF.keys())[i]][f"valid/{conf.bestKey}"]) if mode == 'min' else max(trialDF[list(trialDF.keys())[i]][f"valid/{conf.bestKey}"])
                    if (mode == 'min' and currMetric < bestMetric) or (mode == 'max' and currMetric > bestMetric):
                        bestMetric = currMetric
                        tunePath = list(trialDF.keys())[i]
                except KeyError:
                    print(f"Skipped: {list(trialDF.keys())[i]}")
            conf = conf.copy(analysis.get_all_configs()[tunePath])

    model = modelPackage.Model(conf)
    model = model.to(device)

    if conf.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)
    elif conf.optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)

    return model, optim

def summary(conf, model, dataloaders=None):
    if dataloaders:
        batch = next(iter(dataloaders['test']))
        shape = batch['y'][0].shape
    else:
        shape = (conf.numT + 1, conf.dimY)
    torchsummary.summary(model, shape)

def processData(conf):
    if conf.datasetType == 'classic':
        if conf.customValidTest:
            retLoaders, retDatasets = {}, {}
            for s in ['train', 'valid', 'test']:
                datasetType = conf.customValidTest[s].split(".")[-1]
                loader_func = loaders.npz if datasetType == "npz" else loaders.hdf5
                retLoaders[s], retDatasets[s] = loader_func(conf, customSet=s)
            return retLoaders, retDatasets
        else:
            datasetType = conf.dataset.split(".")[-1]
            loader_func = loaders.npz if datasetType == "npz" else loaders.hdf5
            return loader_func(conf)
    elif conf.datasetType == 'multiCollapse':
        return loaders.multiCollapse(conf)
    else:
        raise ValueError(f"datasetType {conf.datasetType} not valid")

def predict(conf, model, dataloader, epoch, toSave=True, toReturn=False):
    device = next(model.parameters()).device
    model.eval()
    predictions = {set: defaultdict(list) for set in dataloader}

    for set in dataloader:
        with torch.no_grad():
            for data in dataloader[set]:
                true = {k: data[k].to(device) for k in data}
                pred = model(true)
                for k in true:
                    predictions[set][k].extend(list(true[k].cpu().detach().numpy()))
                predictions[set]['pred'].extend(list(pred['y'].cpu().detach().numpy()))

        for k in predictions[set]:
            predictions[set][k] = np.array(predictions[set][k])

        if conf.normalizeY:
            minMax = conf.multiCollapseMinMax
            if conf.fixedB:
                minMax = (minMax[0], minMax[1], minMax[3])
            for i in range(predictions[set]['pred'].shape[1]):
                predictions[set]['pred'][:, i] = predictions[set]['pred'][:, i] * (minMax[i][1] - minMax[i][0]) + minMax[i][0]

        if toSave:
            if not os.path.exists(os.path.join(filesPath(conf), "predictions")):
                os.makedirs(os.path.join(filesPath(conf), "predictions"))
            filePred = os.path.join(filesPath(conf), "predictions", f"{conf.modelLoad.split('.')[0]}-{epoch}-{set}{f'-{conf.filePredAppendix}' if conf.filePredAppendix else ''}.npz")
            np.savez_compressed(filePred, **predictions[set])
            if not conf.has("nonVerbose") or not conf.nonVerbose:
                print(f"Saved {filePred}", flush=True)

    if toReturn:
        return predictions

def evaluate(conf, model, dataloader):
    device = next(model.parameters()).device
    lossFun = MyLoss(conf, device)
    slossFun = SLoss(conf, device)
    model.eval()

    runningLoss = 0.
    runningMetrics = defaultdict(float)

    for data in dataloader:
        true = {k: data[k].to(device) for k in data}
        pred = model(true)
        loss = lossFun(pred['y'], true['y'])
        runningLoss += loss.item()
        if conf.taskType == "classification":
            myPred = torch.softmax(pred['y'], dim=1)
            runningMetrics["acc"] += torch.sum(true['y'].reshape(true['y'].shape[0]) == myPred.max(dim=1).indices).float() / true['y'].shape[0]
        elif conf.taskType == "prediction":
            myPred = torch.exp(pred['y'])
            runningMetrics["kld"] += sp.stats.entropy(true['y'].cpu(), myPred.cpu(), axis=2).sum(axis=1).mean(axis=0)
        elif conf.taskType == "regressionL1":
            runningMetrics["mae"] += loss.item()
            runningMetrics["e"] += slossFun(pred['y'], true['y']).item()
        elif conf.taskType == "regressionL2":
            runningMetrics["mse"] += loss.item()
            runningMetrics["e"] += slossFun(pred['y'], true['y']).item()
        elif conf.taskType == "regressionMRE":
            runningMetrics["mre"] += loss.item()
            runningMetrics["e"] += slossFun(pred['y'], true['y']).item()
        elif conf.taskType == "regressionS":
            runningMetrics["e"] += loss.item()
        else:
            raise ValueError(f"taskType {conf.taskType} not valid")

    for k in runningMetrics:
        runningMetrics[k] /= len(dataloader)

    return runningLoss / len(dataloader), runningMetrics

def runTrain(conf, model, optim, dataloaders, startEpoch, bestValidMetric=None):
    for batchIndex, data in enumerate(trainDataloader):
        true = {k: data[k].to(device) for k in data}
        print("Data shape:", true['x'].shape)  # Add this line to print data shape
        model.zero_grad()

        pred = model(true)

        loss = lossFun(pred['y'], true['y'])
        loss.backward()
        optim.step()

    trainDataloader = dataloaders['train']
    validDataloader = dataloaders['valid']
    testDataloader = dataloaders['test']
    device = next(model.parameters()).device
    lossFun = MyLoss(conf, device)

    if conf.tensorBoard:
        writer = SummaryWriter(os.path.join(filesPath(conf), "tensorBoard"), flush_secs=10)

    if conf.earlyStopping is not None:
        maxMetric = 0.
        currPatience = 0.

    if not os.path.exists(os.path.join(filesPath(conf), "models")):
        os.makedirs(os.path.join(filesPath(conf), "models"))

    if conf.bestSign not in ['<', '>']:
        raise ValueError(f"bestSign {conf.bestSign} not valid")

    if bestValidMetric is None:
        bestValidMetric = -math.inf if conf.bestSign == '>' else math.inf

    globalBatchIndex = 0

    for epoch in range(startEpoch, startEpoch + conf.epochs):
        startTime = datetime.now()
        if not conf.has("nonVerbose") or not conf.nonVerbose:
            print(f"epoch {epoch}", end='', flush=True)

        model.train()
        for data in trainDataloader:
            true = {k: data[k].to(device) for k in data}
            model.zero_grad()
            pred = model(true)
            loss = lossFun(pred['y'], true['y'])
            loss.backward()
            optim.step()

            if conf.logEveryBatch:
                validLoss, validMetrics = evaluate(conf, model, validDataloader)
                testLoss, testMetrics = evaluate(conf, model, testDataloader)

                writerDictLoss = {
                    'train': loss.item(),
                    'valid': validLoss,
                    'test': testLoss,
                }

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if conf.tensorBoard:
                        writer.add_scalars('batch-loss', writerDictLoss, globalBatchIndex)
                globalBatchIndex += 1

        if not conf.has("nonVerbose") or not conf.nonVerbose:
            print(": ", end='', flush=True)

        trainLoss, trainMetrics = evaluate(conf, model, trainDataloader)
        validLoss, validMetrics = evaluate(conf, model, validDataloader)
        testLoss, testMetrics = evaluate(conf, model, testDataloader)

        if conf.useTune:
            tuneDict = {}
            for setStr, lossValue, metricDict in [("train", trainLoss, trainMetrics), ("valid", validLoss, validMetrics), ("test", testLoss, testMetrics)]:
                tuneDict[f"{setStr}/loss"] = lossValue
                for k in metricDict:
                    tuneDict[f"{setStr}/{k}"] = metricDict[k]
            tune.report(**tuneDict)

        if conf.bestKey not in validMetrics:
            raise ValueError(f"bestKey {conf.bestKey} not present")

        writerDictLoss = {
            'train': trainLoss,
            'valid': validLoss,
            'test': testLoss,
        }

        writerDictMetrics = {}
        for k in trainMetrics:
            writerDictMetrics[k] = {
                'train': trainMetrics[k],
                'valid': validMetrics[k],
                'test': testMetrics[k],
            }

        if conf.modelSave != "none":
            fileLast = os.path.join(filesPath(conf), "models", "last.pt")
            if os.path.isfile(fileLast):
                os.remove(fileLast)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'epoch': epoch,
                'bestValidMetric': bestValidMetric,
            }, fileLast)

        if conf.modelSave == "all":
            fileModel = os.path.join(filesPath(conf), "models", f"epoch{epoch}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'epoch': epoch,
                'bestValidMetric': bestValidMetric,
            }, fileModel)
        elif conf.modelSave == "best":
            fileModel = os.path.join(filesPath(conf), "models", "best.pt")
            if (conf.bestSign == '<' and validMetrics[conf.bestKey] < bestValidMetric) or (conf.bestSign == '>' and validMetrics[conf.bestKey] > bestValidMetric):
                bestValidMetric = validMetrics[conf.bestKey]
                if os.path.isfile(fileModel):
                    os.remove(fileModel)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'bestValidMetric': bestValidMetric,
                }, fileModel)

        if conf.logCurves:
            currPath = os.path.join(filesPath(conf), "curves")
            Path(currPath).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(currPath, "epochs.dat"), 'a') as logFile:
                logFile.write(f"{epoch}\n")

            for s in writerDictLoss:
                currPath = os.path.join(filesPath(conf), "curves", "loss", s)
                Path(currPath).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(currPath, "points.dat"), 'a') as logFile:
                    logFile.write(f"{writerDictLoss[s]}\n")

            for k in writerDictMetrics:
                for s in writerDictMetrics[k]:
                    currPath = os.path.join(filesPath(conf), "curves", k, s)
                    Path(currPath).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(currPath, "points.dat"), 'a') as logFile:
                        logFile.write(f"{writerDictMetrics[k][s]}\n")

        if conf.tensorBoard:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                writer.add_scalars('loss', writerDictLoss, epoch)
                for k in writerDictMetrics:
                    writer.add_scalars(f'{k}', writerDictMetrics[k], epoch)

        endTime = datetime.now()
        if not conf.has("nonVerbose") or not conf.nonVerbose:
            print(f"tr. {conf.bestKey} {trainMetrics[conf.bestKey]:.3f} - va. {conf.bestKey} {validMetrics[conf.bestKey]:.3f} - te. {conf.bestKey} {testMetrics[conf.bestKey]:.3f} ({str(endTime - startTime).split('.', 2)[0]}; exp. {str((endTime - startTime) * (startEpoch + conf.epochs - 1 - epoch)).split('.', 2)[0]})", flush=True)

        if conf.earlyStopping is not None:
            if (conf.bestSign == '<' and validMetrics[conf.bestKey] > maxMetric) or (conf.bestSign == '>' and validMetrics[conf.bestKey] < maxMetric):
                if currPatience >= conf.earlyStopping:
                    break
                currPatience += 1
            else:
                maxMetric = validMetrics[conf.bestKey]
                currPatience = 0

    time.sleep(30)

def loadModel(conf, device):
    fileToLoad = os.path.join(filesPath(conf), "models", conf.modelLoad)
    if not conf.has("nonVerbose") or not conf.nonVerbose:
        print(f"Loading {fileToLoad}", flush=True, end="")
    model, optim = makeModel(conf, device)
    checkpoint = torch.load(fileToLoad, map_location=torch.device('cpu'))
    if not conf.has("nonVerbose") or not conf.nonVerbose:
        print(f" (epoch {checkpoint['epoch']})", flush=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optim.load_state_dict(checkpoint['optim_state_dict'])
    return model, optim, checkpoint['epoch'], checkpoint['bestValidMetric']

def getLearningCurves(conf, metric='accuracy'):
    import tensorflow as tf
    sets = ['train', 'valid', 'test']
    path = {s: os.path.join(filesPath(conf), "tensorBoard", metric, s) for s in sets}
    logFiles = {s: list(map(lambda f: os.path.join(path[s], f), sorted(os.listdir(path[s])))) for s in sets}
    values = {s: [v.simple_value for f in logFiles[s] for e in tf.compat.v1.train.summary_iterator(f) for v in e.summary.value if v.tag == 'loss' or v.tag == 'accuracy'] for s in sets}
    return values

def getMaxValidEpoch(conf):
    values = getLearningCurves(conf)
    return np.argmax(values['valid'])

def getMaxValid(conf):
    try:
        values = getLearningCurves(conf)
        return values['valid'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def getMaxTest(conf):
    try:
        values = getLearningCurves(conf)
        return values['test'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def alreadyLaunched(conf):
    return os.path.isdir(os.path.join(filesPath(conf), "tensorBoard"))
