#!/usr/bin/env python

# Examples to run parallel multiple configurations:
# for n in {0..8}; do for c in {0..2}; do ./runPytorch.py configMC1all\[$n\]\[$c\]& done; done
# for n in {0..4}; do for c in {0..2}; do ./runPytorch.py configMC1all\[$n\]\[$c\]& done; done; wait; for n in {5..8}; do for c in {0..2}; do ./runPytorch.py configMC1all\[$n\]\[$c\]& done; done; wait
# kill `ps -e|grep runPyt|cut -d ' ' -f2`

import os
import sys
import socket
import configurations
import funPytorch as fun
import notifier
from datetime import datetime
from loaders import load_data

if __name__ == '__main__':
    timeFormat = "%Y/%m/%d - %H:%M:%S"
    
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Use {} configName [cpu]".format(sys.argv[0]))
    else:
        device = "cpu"  # Set to CPU as default
        
        conf = eval('configurations.{}'.format(sys.argv[1]))
        startTime = datetime.now()
    
        print("====================")
        print("RUN USING {} on device {}".format(sys.argv[1], device))
        print(startTime.strftime(timeFormat))
        print("====================")
        startEpoch = conf.startEpoch
        if startEpoch == 0:
            if os.path.exists(fun.filesPath(conf)) and not (conf.has("force") and conf.force==True):
                raise(Exception("ALREADY LAUNCHED"))
            print("======= CREATE MODEL")
            model, optim = fun.makeModel(conf, device)
            bestValidMetric = None
        else:
            print("======= LOAD MODEL")
            model, optim, loadEpoch, bestValidMetric = fun.loadModel(conf, device)
            if startEpoch == -1:
                startEpoch = loadEpoch + 1
        print("======= LOAD DATA")
        # dataloaders, _ = fun.processData(conf)
        data_loader = load_data()
        data_loader = {'train': data_loader, 'valid': data_loader, 'test': data_loader}  # Update this as needed
        print("======= TRAIN MODEL")
        fun.runTrain(conf, model, optim, data_loader, startEpoch, bestValidMetric)
    
        endTime = datetime.now()
        print("=======")
        print("End {}".format(sys.argv[1]))
        print(endTime.strftime(timeFormat))
        print("====================")
        notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()), 
                             "Start:\t{}\nEnd:\t{}\nDuration:\t{}".format(startTime.strftime(timeFormat),
                                                                         endTime.strftime(timeFormat),
                                                                         str(endTime - startTime).split('.', 2)[0]))
