import pytest
import uqpylab.sessions as sessions
import numpy as np




def model(X):
    return X*np.sin(X)


mySession = sessions.cloud(host="http://localhost:8000", token="636fd25c8c0aa05e344c0cc66abdcfb8c3af73d8")
# (Optional) Get a convenient handle to the command line interface
uq = mySession.cli
# Reset the session
mySession.reset()


uq.rng(100, 'twister')

ModelOpts = {
    'Type': 'Model',
    'ModelFun': 'uqpylab.test.true_models.branin',
    'Parameters': {
        'a': 1,
        'b': 5.1 / (2 * np.pi) ** 2,
        'c': 5 / np.pi,
        'r': 6,
        's': 10,
        't': 1 / (8 * np.pi),
    }
}    
myModel = uq.createModel(ModelOpts)
InputOpts = {"Marginals": [uq.Marginals(M=1, Type="Uniform", Parameters=[-5,10])]+ 
                            [uq.Marginals(M=1, Type="Uniform", Parameters=[0,15])]}
myInput = uq.createInput(InputOpts)
X = uq.getSample(myInput, 10, 'Sobol')
Y = uq.evalModel(myModel, X)
SeqPCKOpts = {
    "Type": "Metamodel",
    "MetaType": "PCK",
    "Mode": "Sequential",
    "ExpDesign": {
        "X": X.tolist(),
        "Y": Y.tolist(),
    },
}
mySeqPCK = uq.createModel(SeqPCKOpts)
uq.print(mySeqPCK)
fig = uq.display(mySeqPCK,test_mode=True);    
assert len(fig) == 1    
