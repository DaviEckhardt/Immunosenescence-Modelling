import pytest
import uqpylab.sessions as uq_session
import numpy as np
import time


def test_common_scenarios(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    # Construct an input object with various input distributions
    iOptsTrue  = {
       'Marginals': [
        {
            'Type': 'Gaussian',
            'Parameters': [0,1],
            # 'Bounds': [1,10]
        },
        {
            'Type': 'Beta',
            'Parameters': [6,4],
        }
    ]

        }
    print("Creating the true input...")
    myInputTrue = uq.createInput(iOptsTrue, request_format='JSON', response_format='MAT')
    print("Done.")
    print("Generating samples...")
    X = uq.getSample(myInputTrue,N=500,Method='LHS', request_format='JSON', response_format='MAT')
    print("Done.")
    InputOpts = {
    "Copula" : {
        "Type": "Independent"
        }
    }
    InputOpts["Inference"] = {
        "Data": X.tolist()
    }

    print("Performing Inference...")
    M = 2
    InputOpts["Marginals"] = [{"Type": "auto"} for i in range(M)]
    InputOpts["Marginals"][0]["Inference"] =  {"Criterion": "KS"}
    InputOpts["Marginals"][0]["Type"] = ['Gaussian', 'Uniform', 'Beta']
    InputHat2 = uq.createInput(InputOpts, request_format='JSON', response_format='MAT')

    print("Validating results...")
    for idx, m in enumerate(myInputTrue['Marginals']):
        print(f"{m['Type']} vs {InputHat2['Marginals'][idx]['Type']}")
        print(f"{m['Parameters']} vs {InputHat2['Marginals'][idx]['Parameters']}")
        assert m['Type'] == InputHat2['Marginals'][idx]['Type']
        p1 = np.array(m['Parameters'])
        p2 = np.array(InputHat2['Marginals'][idx]['Parameters'])
        assert np.max(np.abs(p1 - p2)) < 5e-2

def test_non_parametric_marginals_inference(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    iOptsTrue = {
    'Marginals': [
        {
            'Type': 'Gaussian',
            'Parameters': [0,1],
            'Bounds': [1,10]
        },
        {
            'Type': 'Beta',
            'Parameters': [6,4],
        }
            ]
        }

    myInputTrue = uq.createInput(iOptsTrue)

    X = uq.getSample(myInputTrue,10)

    InputOpts = {
        "Copula" : {
            "Type": "Independent"
        },
        "Inference": {
            "Data": X.tolist()
        }
    }

    M = 2
    InputOpts["Marginals"] = [{"Type": "auto"} for i in range(M)]

    InputOpts["Marginals"][1]['Type'] =  "ks" 

    InputOpts["Marginals"][1]["Options"] = {
        "Kernel": "triangle",
        "Bandwidth": 0.1
    }

    InputHat = uq.createInput(InputOpts)
    assert 'GoF' in InputHat['Marginals'][0]
    assert 'GoF' in InputHat['Marginals'][1]
    assert 'KS'  in InputHat['Marginals'][1]

def test_gaussian_copula_inference(request, helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    X = np.random.rand(200,3)

    iOpts = {
        'Inference': {'Data': X.tolist()},
        'Copula': {
                'Type': 'Gaussian'
            }
    }

    InputHat = uq.createInput(iOpts)
    assert 'Inference' in InputHat['Copula']
    assert 'GoF' in InputHat['Copula']
    assert 'Criterion' in InputHat['Copula']['Inference']
    assert 'PairIndepTest' in InputHat['Copula']['Inference']
    assert 'Alpha' in InputHat['Copula']['Inference']['PairIndepTest']
    assert 'Type' in InputHat['Copula']['Inference']['PairIndepTest']
    
def test_Inference(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # generate data
    iOptsTrue = {'Marginals': uq.StdNormalMarginals(1) + [uq.Marginals(1, "Beta", [6,4])]}
    iOptsTrue['Marginals'][0]['Bounds'] = [1,10]
    myInputTrue = uq.createInput(iOptsTrue)
    X = uq.getSample(myInputTrue,100)

    # inference of marginals
    InputOpts = {
        "Inference": {"Data": X.tolist()},
        "Marginals": [{"Type": "auto"} for i in range(2)],
    }
    # full inference with Kolmogorov-Smirnov selection criterion for the second input marginal
    InputOpts["Marginals"][1]["Inference"] = {"Criterion": 'KS'}
    InputHat1 = uq.createInput(InputOpts)
    X1 = uq.getSample(InputHat1,50)
    assert X1.shape == (50,2)
    # assign data to the second marginal and get samples
    InputOpts['Marginals'][1]['Inference']['Data'] = X[:,1].tolist()
    InputHat2 = uq.createInput(InputOpts)
    X2 = uq.getSample(InputHat2,50)
    assert X2.shape == (50,2)  
    # constrained set of marginal families
    InputOpts["Marginals"][0]["Type"] = ["Gaussian", "Exponential", "Weibull"]
    InputHat3 = uq.createInput(InputOpts)
    X3 = uq.getSample(InputHat3,50)  
    assert X3.shape == (50,2)    
    # full inference with truncated marginal
    InputOpts["Marginals"][0]["Bounds"] = [1, 10]
    InputHat4 = uq.createInput(InputOpts)
    X4 = uq.getSample(InputHat4,50)  
    assert X4.shape == (50,2)   
    InputOpts["Marginals"][0]["Bounds"] = [1, float('inf')]
    InputHat4a = uq.createInput(InputOpts)
    X4a = uq.getSample(InputHat4a,50)  
    assert X4a.shape == (50,2) 
    # parameter fitting of a fixed marginal family
    InputOpts["Marginals"][0]["Type"] = "Gaussian"
    InputOpts['Marginals'][0]['Parameters'] = [0,1]
    InputHat5 = uq.createInput(InputOpts)
    X5 = uq.getSample(InputHat5,50)  
    assert X5.shape == (50,2)
    # inference by kernel smoothing
    InputOpts["Marginals"][1]['Type'] =  "ks" 
    InputOpts["Marginals"][1]["Options"] = {
        "Kernel": "Epanechnikov", # Gaussian, Normal, Triangle, Triangular, Box, Epanechnikov
        # "Bandwidth": 0.1
    }
    InputHat6 = uq.createInput(InputOpts)
    X6 = uq.getSample(InputHat6,50)  
    assert X6.shape == (50,2)    
    # inference of selected marginals
    InputOpts["Marginals"][0] = {
        "Type": "Gaussian",
        "Parameters": [0, 1],
        "Bounds": [1,10]
    }
    InputHat7 = uq.createInput(InputOpts)
    X7 = uq.getSample(InputHat7,50)  
    assert X7.shape == (50,2)     
    # specification of inference options for each marginal
    del InputOpts
    InputOpts = {
        'Marginals': [
            {
                'Type': 'auto',
                'Inference': {
                    'Criterion': 'BIC',
                    'Data': X[:,0].tolist()
                }
            },
            {
                'Type': 'Beta',
                'Inference': {
                    'Data': X[:,1].tolist()
                }
            }
        ],
        
    }
    InputHat8 = uq.createInput(InputOpts)    
    X8 = uq.getSample(InputHat8,50)  
    assert X8.shape == (50,2)

def test_InferenceCopula(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # data generation
    iOptsTrue = {'Marginals': [uq.Marginals(1,'Gaussian',[-1,1])] + [uq.Marginals(1,'Exponential',[1])] + [uq.Marginals(1, 'Uniform',[-1,3])]}
    iOptsTrue['Copula'] = {
        'Type': 'CVine',
        'Structure': [[3,1,2]],
        'Families': ['Gaussian', 'Gumbel', 't'],
        'Rotations': [0,90,0],
        'Parameters': [.4, 2, [-.2, 2]]
    }
    myInputTrue = uq.createInput(iOptsTrue)
    X = uq.getSample(myInputTrue,100)
    assert X.shape == (100,3)
    U = uq.all_cdf(X,myInputTrue['Marginals'])
    assert U.shape == (100,3)
    # Inference of marginals and copula
    iOpts = {'Inference': {'Data': X.tolist()}}
    InputHat1 = uq.createInput(iOpts)
    X1 = uq.getSample(InputHat1,100)
    assert X1.shape == (100,3)
    # Testing for block independence
    iOpts['Inference']['BlockIndepTest'] = {
        'Alpha': 0.05,
        'Type': 'Kendall', # Kendall, Spearman, Pearson
        'Correction': 'auto' # none, fdr, Bonferroni, auto
    }
    InputHat2a = uq.createInput(iOpts)
    X2a = uq.getSample(InputHat2a,100)
    assert X2a.shape == (100,3)
    # Turn the block independence test
    iOpts['Inference']['BlockIndepTest']['Alpha'] = 0
    InputHat2b = uq.createInput(iOpts)   
    X2b = uq.getSample(InputHat2b,100)
    assert X2b.shape == (100,3)
    # Block independence test options can be provided as copula inference options rather than general options
    iOpts['Copula'] = {
        'Type': 'auto',
        'Inference': {
            'BlockIndepTest':
            {
                'Alpha': 0.05
            }
        }
    }
    InputHat2c = uq.createInput(iOpts)
    X2c = uq.getSample(InputHat2c,100)
    assert X2c.shape == (100,3)
    # Specify data for copula inference
    iOpts['Copula']['Inference']['Data'] = X.tolist()
    InputHat3a = uq.createInput(iOpts)
    X3a = uq.getSample(InputHat3a,100)
    assert X3a.shape == (100,3)
    # Specify the data for data as pseudo-observations in [0,1]<sup>M</sup>
    del iOpts['Copula']['Inference']['Data']
    iOpts['Copula']['Inference']['DataU'] = U.tolist()
    InputHat3b = uq.createInput(iOpts)
    X3b = uq.getSample(InputHat3b,100)
    assert X3b.shape == (100,3)   
    # Inference among a selected list of copula types
    iOpts['Copula']['Type'] = ['DVine', 'CVine']
    InputHat4 = uq.createInput(iOpts)
    X4 = uq.getSample(InputHat4,100)
    assert X4.shape == (100,3)      
    # Testing for pair independence
    iOpts['Copula']['Inference'] = {
        'PairIndepTest': {
            'Type': 'Pearson', # Kendall, Spearman, Pearson
            'Alpha': 0.05
        }
    }
    InputHat5a = uq.createInput(iOpts)
    X5a = uq.getSample(InputHat5a,100)
    assert X5a.shape == (100,3)
    # set Bonferroni correction
    iOpts['Copula']['Inference']['PairIndepTest']['Correction'] = 'Bonferroni' # auto, none, fdr, Bonferroni
    InputHat5b = uq.createInput(iOpts)
    X5b = uq.getSample(InputHat5b,100)
    assert X5b.shape == (100,3)

    # Different selection criteria for marginals and copula inference
    iOpts['Inference']['Criterion'] = 'BIC' # AIC, ML, BIC, KS
    InputHat6 = uq.createInput(iOpts)
    X6 = uq.getSample(InputHat6,100)
    assert X6.shape == (100,3)
    # Copula ionference with fixed copula type
    # Gaussian copula
    iOpts['Copula'] = {'Type': 'Gaussian'}
    InputHat7a = uq.createInput(iOpts)
    X7a = uq.getSample(InputHat7a,100)
    assert X7a.shape == (100,3)
    # Vine copulas
    iOpts['Copula'] = {
        'Type': 'CVine',
        'Inference': {
            'CVineStructure': [[3,1,2]],
            'PCFamilies': ['Gaussian', 'Gumbel', 't']
        }
    }
    InputHat7b = uq.createInput(iOpts)
    X7b = uq.getSample(InputHat7b,100)
    assert X7b.shape == (100,3)
    # # Marginals inference with fully specified copula
    iOpts['Copula'] = {'Type': 'Gaussian'}
    iOpts['Copula']['Parameters'] = [[1, -.4, .3], [-.4, 1, -.6], [.3, -.6, 1]]
    InputHat8 = uq.createInput(iOpts)
    X8 = uq.getSample(InputHat8,100)
    assert X8.shape == (100,3)
    # # Copula inference with fixed marginals
    del iOpts
    iOpts = {
        'Marginals': [uq.Marginals(1,'Gaussian',[-1,1])] + [uq.Marginals(1, 'Exponential', [1])] + [uq.Marginals(1,'Uniform',[-1,3])],
        'Inference': {'Data': X.tolist()} 
    }
    InputHat9 = uq.createInput(iOpts)
    X9 = uq.getSample(InputHat9,100)
    assert X9.shape == (100,3)
    # Copula inference using different data
    iOpts['Copula'] = {
        'Type': 'auto',
        'Inference': {'Data': X[::2].tolist()}
    }
    InputHat10 = uq.createInput(iOpts)
    X10 = uq.getSample(InputHat10,100)
    assert X10.shape == (100,3)  
    # Copula inference on pseudo-observations in the unit hypercube 
    del iOpts
    iOpts = {
        'Marginals': [uq.Marginals(1,'Gaussian',[-1,1])] + [uq.Marginals(1,'Exponential',[1])] + [uq.Marginals(1,'Uniform',[-1,3])],
        'Copula': {
            'Type': 'auto',
            'Inference': {'DataU': U.tolist()}
        }
    }
    InputHat11 = uq.createInput(iOpts)
    X11 = uq.getSample(InputHat11,100)
    assert X11.shape == (100,3)

def test_InferencePairCopula(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')

    # data generation
    iOptsTrue = {'Marginals': [uq.Marginals(1,'Gaussian',[-1,1])] + [uq.Marginals(1,'Exponential',[1])] + [uq.Marginals(1, 'Uniform',[-1,3])]}
    iOptsTrue['Copula'] = {
        'Type': 'CVine',
        'Structure': [[3,1,2]],
        'Families': ['Gaussian', 'Gumbel', 't'],
        'Rotations': [0,90,0],
        'Parameters': [.4, 2, [-.2, 2]]
    }
    myInputTrue = uq.createInput(iOptsTrue)
    X = uq.getSample(myInputTrue,100)
    assert X.shape == (100,3)

    # Inference of pair copulas
    iOpts = {
        'Inference': {'Data': X[:,:2].tolist()},
        'Copula': {'Type': 'Pair'}
    }
    InputHat5c = uq.createInput(iOpts)
    X5c = uq.getSample(InputHat5c,100)
    assert X5c.shape == (100,2)
    # select among all supported pair copula families
    iOpts['Copula']['Inference'] = {
        'PCfamilies' : ['Gaussian', 'Frank', 'Clayton']
    }
    InputHat5d = uq.createInput(iOpts)
    X5d = uq.getSample(InputHat5d,100)
    assert X5d.shape == (100,2)    
    # infer the copula on data Xnew
    iOptsTrue2 = {'Marginals': uq.StdUniformMarginals(2),
                  'Copula': {'Type': 'Pair','Family': 'Gumbel','Parameters': [1.5]}
                 }
    myInputTrue2 = uq.createInput(iOptsTrue2)
    Xnew = uq.getSample(myInputTrue2,100)
    iOpts['Copula']['Inference']['Data'] = Xnew.tolist()
    InputHat5e = uq.createInput(iOpts)
    X5e = uq.getSample(InputHat5e,100)
    assert X5e.shape == (100,2)
    # presudo-observations U directly provided for copula inference
    Unew = uq.all_cdf(Xnew,myInputTrue2['Marginals'])
    del iOpts['Copula']['Inference']['Data']
    iOpts['Copula']['Inference']['DataU'] = Unew.tolist()
    InputHat5f = uq.createInput(iOpts) 
    X5f = uq.getSample(InputHat5f,100)
    assert X5f.shape == (100,2)
    mySession.quit()
    time.sleep(1)    
