import pytest
import numpy as np
import time

def test_discrepancy(request,helpers):
    print('Running tests for Bayesian inversion...')
    print('...Single model with scalar model output...')
    mySession = helpers.init_session(request)
    uq = mySession.cli
    uq.rng(100, 'twister')
    ModelOpts = {
        'Type' : 'Model',
        'Name': 'Forward model',
        'mString': '5/32*X(5)*X(3)^4/(X(4)*X(1)*X(2)^3)'
    }

    myModel = uq.createModel(ModelOpts)

    PriorOpts = {
    "Marginals": [
        {
        "Name": "b", # beam width
        "Type": "Constant",
        "Parameters": [0.15] # (m)
        },
        {
        "Name": "h", # beam height
        "Type": "Constant",
        "Parameters": [0.3] # (m)
        },
        {
        "Name": "L", # beam length
        "Type": "Constant",
        "Parameters": [5] # (m) 
        },
        {
        "Name": "E", # Young's modulus
        "Type": "LogNormal",
        "Moments": [30e9,4.5e9] # (N/m^2)
        },
        {
        "Name": "p", # constant distributed load   
        "Type": "Constant",
        "Parameters": [12000] # (N/m)
        }
    ]
    }

    myPriorDist = uq.createInput(PriorOpts)

    V_mid = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000 # (m)
    myData = {
        'y': V_mid.tolist(),
        'Name': 'Beam mid-span deflection',
    }

    Solver = {
        'Type': 'MCMC',
        'MCMC': {
            'Sampler': 'AM',
            'Steps': 1e2, # UQLab current setting: 1e3
            'NChains': 1e2,
            'TO': 1e2,
            'Proposal': {
                'PriorScale': 0.1
            }
        }
    }    

    BayesOpts = {
        "Type" : "Inversion",
        "Data" : myData,
        'Solver': Solver,
    }

    print('...Discrepancy model with known residual variance...')

    DiscrepancyOpts = {
        'Type': 'Gaussian',
        'Parameters': [1e-7], # (m^2),
    }

    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Prior'] = myPriorDist['Name']

    myBayesianAnalysis2a = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis2a)
    uq.display(myBayesianAnalysis2a,test_mode=True)
    print('Done ...')

    print('...Discrepancy model with unknown residual variance...')
    DiscrepancyPriorOpts = {
        'Name': 'Prior of discrepancy parameter',
        'Marginals': {
            'Name': 'Sigma2',
            'Type': 'Uniform',
            'Parameters': [0, np.mean(V_mid)**2]
        }
    }

    myDiscrepancyPrior = uq.createInput(DiscrepancyPriorOpts)

    DiscrepancyOpts = {
        'Type': 'Gaussian',
        'Prior': myDiscrepancyPrior['Name']
    }

    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Prior'] = myPriorDist['Name'] 

    myBayesianAnalysis2b = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis2b)
    uq.display(myBayesianAnalysis2b,test_mode=True)    
    print('Done ...')

def test_multiple_model_outputs(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli

    ModelOpts = {
        'Type' : 'Model',
        'Name': 'Forward model',
        'mString': '[(57/512)*(X(:, 5).*X(:, 3).^4)./(X(:, 4).*X(:, 1).*X(:, 2).^3), 5/32.*X(:, 5).*X(:, 3).^4./(X(:, 4).*X(:, 1).*X(:, 2).^3)]'
    }    
    myModel = uq.createModel(ModelOpts)

    PriorOpts = {
        "Marginals": [
            {
                "Name": "b", # beam width
                "Type": "Constant",
                "Parameters": [0.15] # (m)
            },
            {
                "Name": "h", # beam height
                "Type": "Constant",
                "Parameters": [0.3] # (m)
            },
            {
                "Name": "L", # beam length
                "Type": "Constant",
                "Parameters": [5] # (m) 
            },
            {
                "Name": "E", # Young's modulus
                "Type": "LogNormal",
                "Moments": [30e9,4.5e9] # (N/m^2)
            },
            {
                "Name": "p", # constant distributed load   
                "Type": "Constant",
                "Parameters": [12000] # (N/m)
            }
        ]
    }
    myPriorDist = uq.createInput(PriorOpts)
    V = np.array([[8.98, 8.66, 8.85, 9.19, 8.64],            # L/4 (m)
                  [12.84, 13.12, 12.13, 12.19, 12.67]])/1000 # L/2 (m)
    myData = {
        'y': V.T.tolist(),
        'Name': 'Beam quarter and midspan deflection',
    }
    BayesOpts = {
        "Type" : "Inversion",
        "Data" : myData,
        "Solver": {
            "Type": "MCMC",
            "MCMC": {
                "Steps": 20
            }
        }
    }
    print("... Multiple model outputs...")
    myBayesianAnalysis = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis)
    uq.display(myBayesianAnalysis,test_mode=True)  
    print("Done...")

    print("... And testing the discrepancy models for multiple model outputs...")
    print("... with known residual variance, independent and identically distributed eps_i (Parameters is a scalar)...")
    DiscrepancyOpts = {
        'Type': 'Gaussian',
        'Parameters': 1e-7
    }
    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Prior'] = myPriorDist['Name']
    myBayesianAnalysis2a = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis2a)
    uq.display(myBayesianAnalysis2a,test_mode=True)      
    print("Done...")

    print("... with known residual variance, independent eps_i (Parameters is a list)...")
    DiscrepancyOpts = {
        'Type': 'Gaussian',
        'Parameters': [[1e-7, 5e-7]] # row vector of length N_out
    }

    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Prior'] = myPriorDist['Name']

    myBayesianAnalysis2b = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis2b)
    uq.display(myBayesianAnalysis2b,test_mode=True)      
    print("Done...")

    print("... with known residual variance, dependent eps_i (Parameters is a nested list)...")
    DiscrepancyOpts = {
        'Type': 'Gaussian',
        'Parameters':  [[1e-7, -5e-8],[-5e-8, 5e-7]] # N_out x N_out matrix
    }

    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Prior'] = myPriorDist['Name']

    myBayesianAnalysis2c = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis2c)
    uq.display(myBayesianAnalysis2c,test_mode=True)      
    print("Done...")

    print("... with UNknown residual variance, independent and identically distributed eps_i...")
    DiscrepancyPriorOpts = {
        'Marginals': {
            'Name': 'Sigma2',
            'Type': 'Uniform',
            'Parameters': [[0, 1e-4]] # (m^2),
        }
    }
    myDiscrepancyPrior = uq.createInput(DiscrepancyPriorOpts)
    DiscrepancyOpts = {
        'Type': 'Gaussian',
        'Prior': myDiscrepancyPrior['Name']
    }
    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Prior'] = myPriorDist['Name']
    myBayesianAnalysis3a = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis3a)
    uq.display(myBayesianAnalysis3a,test_mode=True)      
    print("Done...")

    print("... with UNknown residual variance, independent eps_i...")
    DiscrepancyPriorOpts = {
        'Name': 'Prior of discrepancy parameter',
        'Marginals': [
            {
                'Name': 'Sigma2_1',
                'Type': 'Lognormal',
                'Moments': [1e-5, 5e-6] # (m^2)
            },
            {
                'Name': 'Sigma2_2',
                'Type': 'Uniform',
                'Parameters': [0, 1e-4] # (m^2)
            }
        ]
    }
    myDiscrepancyPrior = uq.createInput(DiscrepancyPriorOpts)
    DiscrepancyOpts = {
        'Type': 'Gaussian',
        'Prior': myDiscrepancyPrior['Name']
    }
    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Prior'] = myPriorDist['Name']
    myBayesianAnalysis3b = uq.createAnalysis(BayesOpts)   
    uq.print(myBayesianAnalysis3b)
    uq.display(myBayesianAnalysis3b,test_mode=True)      
    print("Done...")

def test_data_and_discrepancy_groups(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    print("...testing data and discrepancy groups")
    ModelOpts = {
        'Type' : 'Model',
        'mString': '[(57/512)*(X(:, 5).*X(:, 3).^4)./(X(:, 4).*X(:, 1).*X(:, 2).^3), 5/32.*X(:,5).*X(:,3).^4./(X(:,4).*X(:,1).*X(:,2).^3)]',
        'isVectorized': 1
    }
    myModel = uq.createModel(ModelOpts)

    PriorOpts = {
        "Marginals": [
            {
                "Name": "b", # beam width
                "Type": "Constant",
                "Parameters": [0.15] # (m)
            },
            {
                "Name": "h", # beam height
                "Type": "Constant",
                "Parameters": [0.3] # (m)
            },
            {
                "Name": "L", # beam length
                "Type": "Constant",
                "Parameters": [5] # (m) 
            },
            {
                "Name": "E", # Young's modulus
                "Type": "LogNormal",
                "Moments": [30e9,4.5e9] # (N/m^2)
            },
            {
                "Name": "p", # constant distributed load   
                "Type": "Constant",
                "Parameters": [12000] # (N/m)
            }
        ]
    }
    myPriorDist = uq.createInput(PriorOpts)
    BayesOpts = {
        "Type" : "Inversion",
        "Solver": {
            "Type": "MCMC",
            "MCMC": {
                "Steps": 20
            }
        }        
    }
    # group 1
    V_quart = np.array([10.51,  9.60, 10.22,  8.16,  7.47])/1000  # L/4 (m)

    Data = [
        {
            'y': V_quart.tolist(),
            'Name': 'Deflection measurements at L/4',
            'MOMap': [1] # Model Output Map
        }
    ]

    DiscrepancyOpts = [
        {
            'Type': 'Gaussian',
            'Parameters': 1e-7 # (m^2)
        }
    ]

    # group 2
    V_mid = np.array([12.59, 11.23, 15.28, 12.45, 13.21])/1000 # L/2 (m)

    Data.append(
        {
            'y': V_mid.tolist(),
            'Name': 'Deflection measurements at L/2',
            'MOMap': [2] # Model Output Map
        }
    )

    DiscrepancyPriorOpts = {
        # 'Name': 'Prior of sigma',
        'Marginals': {
            'Name': 'Sigma2_2',
            'Type': 'Uniform',
            'Parameters':  [0, 1e-4], # (m^2)
        }
    }
    DiscrepancyPrior = uq.createInput(DiscrepancyPriorOpts)

    DiscrepancyOpts.append(
        {
            'Type': 'Gaussian',
            'Prior': DiscrepancyPrior['Name']
        }
    )
    BayesOpts['Prior'] = myPriorDist['Name']
    BayesOpts['Discrepancy'] = DiscrepancyOpts
    BayesOpts['Data'] = Data

    myBayesianAnalysis4 = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis4)
    uq.display(myBayesianAnalysis4,test_mode=True)      
    print("Done...")

def test_solvers(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli

    ModelOpts = {
        'Type' : 'Model',
        'mString': '5/32*X(5)*X(3)^4/(X(4)*X(1)*X(2)^3)'
    }
    myModel = uq.createModel(ModelOpts)
    PriorOpts = {
        "Marginals": [
            {
                "Name": "b", # beam width
                "Type": "Constant",
                "Parameters": [0.15] # (m)
            },
            {
                "Name": "h", # beam height
                "Type": "Constant",
                "Parameters": [0.3] # (m)
            },
            {
                "Name": "L", # beam length
                "Type": "Constant",
                "Parameters": [5] # (m) 
            },
            {
                "Name": "E", # Young's modulus
                "Type": "LogNormal",
                "Moments": [30e9,4.5e9] # (N/m^2)
            },
            {
                "Name": "p", # constant distributed load   
                "Type": "Constant",
                "Parameters": [12000] # (N/m)
            }
        ]
    }
    myPriorDist = uq.createInput(PriorOpts)
    V_mid = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000 # (m)
    myData = {
        'y': V_mid.tolist(),
        'Name': 'Beam mid-span deflection',
    }
    BayesOpts = {
        "Type" : "Inversion",
        "Data" : myData,
    }
    print("...Markov Chain Monte Carlo...")

    Solver = {
        'Type': 'MCMC',
        'MCMC': {
            'Sampler': 'MH', # AM, HMC, AIES
            'NChains': 100,
        }
    }

    Solver['MCMC']['Steps'] = 20
    BayesOpts['Solver'] = Solver
    PriorDist = {
        'Marginals': [
            {
                'Name': 'Sigma2',
                'Type': 'Uniform',
                'Parameters': [0, np.mean(V_mid)**2]
            },
            {
                'Name': 'E',
                'Type': 'Lognormal',
                'Moments': [30e9, 4.5e9]
            }
        ]
    }

    myPriorDist2 = uq.createInput(PriorDist)
    print("...with specified seeds")
    Seed = uq.getSample(myPriorDist2,Solver['MCMC']['NChains'],'Sobol').T
    Solver['MCMC']['Seed']  = Seed.tolist()
    BayesOpts['Solver'] = Solver
    BayesOpts['Prior'] = myPriorDist['Name']    
    myBayesianAnalysis1a = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis1a)
    uq.display(myBayesianAnalysis1a,test_mode=True)      
    print("Done...")

    print("...MCMC with MH algorithm")
    myProposal = {
        'PriorScale': 0.1
    }
    Solver['Proposal'] = myProposal
    BayesOpts['Solver'] = Solver
    myBayesianAnalysis1b = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis1b)
    uq.display(myBayesianAnalysis1b,test_mode=True)   
    print("Done...")

    print("...and specified covariance matrix...")
    myProposal['Cov'] = [[1,0],[0,1]]
    Solver['Proposal'] = myProposal
    BayesOpts['Solver'] = Solver
    myBayesianAnalysis1c = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis1c)
    uq.display(myBayesianAnalysis1c,test_mode=True)  
    print("Done...")

    print("...advanced proposal distributions...")
    ProposalDistribution = {
        # 'Name': 'Prior of discrepancy parameter 2',
        'Marginals': [
            {
                'Name': 'Sigma2',
                'Type': 'Uniform',
                'Parameters': [0, np.mean(V_mid)**2]
            },
            {
                'Name': 'E',
                'Type': 'Lognormal',
                'Moments': [30e9, 4.5e9]
            }
        ]
    }

    myProposalDistribution = uq.createInput(ProposalDistribution)
    myProposal = {
        'Distribution': myProposalDistribution['Name'],
        'Conditioning': 'Previous' # Other valid option: 'Global'
    }
    Solver['Proposal'] = myProposal
    BayesOpts['Solver'] = Solver
    BayesOpts['Prior'] = myPriorDist['Name']
    myBayesianAnalysis1d = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis1d)
    uq.display(myBayesianAnalysis1d,test_mode=True)      
    print("Done...")

    print("...adaptive Metropolis algorithm...")
    SolverMCMC = {
        'Proposal': myProposal,
        'MCMC': {
            'T0': 1e2,
            'Epsilon':  1e-4,
            'Steps': 20
        }
    }
    BayesOpts['Solver'] = SolverMCMC
    myBayesianAnalysis1e = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis1e)
    uq.display(myBayesianAnalysis1e,test_mode=True)      
    print("Done...")

    print("...Hamiltonian Monte Carlo algorithm...")
    Solver = {
        'MCMC': {
            'LeapfrogStep': 40,
            'LeapfrogSize': 0.1,
            'Mass': 1,
            'Steps': 20
        }
    }

    BayesOpts['Solver'] = Solver
    myBayesianAnalysis1f = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis1f)
    uq.display(myBayesianAnalysis1f,test_mode=True)      
    print("Done...")

    ("...Affine invariant ensemble algorithm...")
    SolverMCMC = {'MCMC': {
        'a': 3,
        'Steps': 20
        }
    }
    BayesOpts['Solver'] = SolverMCMC
    myBayesianAnalysis1g = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis1g)
    uq.display(myBayesianAnalysis1g,test_mode=True)      
    print("Done...")

    print("... Spectral likelihood expansion...")
    Solver = {
        'Type': 'SLE'
    }

    Solver['SLE'] = {
        'Degree': list(range(1,21)),
        'ExpDesign': {
            'NSamples': 1e2
        }
    }

    BayesOpts['Solver'] = Solver
    myBayesianAnalysis2a = uq.createAnalysis(BayesOpts)
    print("Done...")


    Solver = {
        'Type': 'SSLE'
    }

    print("...Stochastic spectral likelihood embedding...")
    Solver['SSLE'] = {
        # Expansion options
        'ExpOptions': {
            'Degree': list(range(1,5))
        },
        # Experimental design options
        'ExpDesign': {
            'NSamples': 20,
            'NEnrich': 10
        }
    }
    BayesOpts['Solver'] = Solver
    myBayesianAnalysis3a = uq.createAnalysis(BayesOpts)    
    print("Done...")

    print("...No solver: Posterior point by point evaluation...")
    BayesOpts['Solver'] = {'Type': 'None'}
    myBayesianAnalysis4a = uq.createAnalysis(BayesOpts)    
    print("Done...")

def test_multiple_forward_models(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli
    print("Testing multiple forward models...")
    PriorOpts = {
        "Marginals": [
            {
                "Name": "b", # beam width
                "Type": "Constant",
                "Parameters": [0.15] # (m)
            },
            {
                "Name": "h", # beam height
                "Type": "Constant",
                "Parameters": [0.3] # (m)
            },
            {
                "Name": "L", # beam length
                "Type": "Constant",
                "Parameters": [5] # (m) 
            },
            {
                "Name": "E", # Young's modulus
                "Type": "LogNormal",
                "Moments": [30e9,4.5e9] # (N/m^2)
            },
            {
                "Name": "p", # constant distributed load   
                "Type": "Constant",
                "Parameters": [12000] # (N/m)
            },
            {
                'Name': 'P',
                'Type': 'Constant',
                'Parameters': [50000] # (N)
            }            
        ]
    }
    myPriorDist = uq.createInput(PriorOpts)
    ModelOpts1 = {
        'Type' : 'Model',
        'Name': 'Beam bending deflection',
        'mString': '5/32.*X(:,5).*X(:,3).^4./(X(:,4).*X(:,1).*X(:,2).^3)',
        'isVectorized': 1
    }
    myModel1 = uq.createModel(ModelOpts1)
    ModelOpts2 = {
        'Type' : 'Model',
        'Name': 'Beam elongation',
        'mString':'X(:,5).*X(:,3)./(X(:,1).*X(:,2).*X(:,4))',
        'isVectorized': 1,
    }
    myModel2 = uq.createModel(ModelOpts2)
    ForwardModels = [
        {
            'Model': myModel1['Name'],
            'PMap': [1,2,3,4,5]
        },
        {
            'Model': myModel2['Name'],
            'PMap': [1,2,3,4,6]

        }
    ]
    V_mid = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000 # (m)
    U_right = np.array([0.235, 0.236, 0.229])/1000             # (m)

    myData = [
        # Data group 1
        {
            'y': V_mid.tolist(),
            'Name': 'Beam mid-span deflection',
            'MOMap': [1,        # Model ID
                    1]        # Output ID
        },
        # Data group 2
        {
            'y': U_right.tolist(),
            'Name': 'Beam elongation',
            'MOMap': [2,        # Model ID
                    1  ]      # Output ID
        }
    ]
    # Discrepancy group 1
    DiscrepancyPriorOpts1 = {
        'Name': 'Prior of sigma_1^2',
        'Marginals': {
            'Name': 'Sigma2',
            'Type': 'Uniform',
            'Parameters':  [0, 1e-4], #(m^2)
        }
    }
    myDiscrepancyPrior1 = uq.createInput(DiscrepancyPriorOpts1)

    DiscrepancyOpts = [
        # Discrepancy group 1
        {
            'Type': 'Gaussian',
            'Prior': myDiscrepancyPrior1['Name']
        },
        # Discrepancy group 2
        {
            'Type': 'Gaussian',
            'Parameters': 2e-11, # (m^2) known discr. variance
        }
    ]
    BayesOpts = {
        "Type": "Inversion",
        "Name": "Bayesian multiple model",
        "Prior": myPriorDist["Name"],
        "ForwardModel": ForwardModels,
        "Data" : myData,
        "Discrepancy": DiscrepancyOpts,
        "Solver": {
            "Type": "MCMC",
            "MCMC": {
                "Steps": 20
            }
        }        
    }
    myBayesianAnalysis = uq.createAnalysis(BayesOpts)
    uq.print(myBayesianAnalysis)
    uq.display(myBayesianAnalysis,test_mode=True)      
    print("Done...")

def test_postProcessInversion(request,helpers):
    mySession = helpers.init_session(request)
    uq = mySession.cli

    ModelOpts = {
        "Type": "Model",
        "mString": "(5/32)*(X(:, 5).*X(:, 3).^4)./(X(:, 4).*X(:, 1).*X(:, 2).^3)",
    }

    myModel = uq.createModel(ModelOpts)

    PriorOpts = {
        "Marginals": [
            {
                "Name": "b",  # beam width
                "Type": "LogNormal",
                "Moments": [0.15, 0.0075],  # (m)
            },
            {
                "Name": "h",  # beam height
                "Type": "LogNormal",
                "Moments": [0.3, 0.015],  # (m)
            },
            {
                "Name": "L",  # beam length
                "Type": "LogNormal",
                "Moments": [5, 0.05],  # (m)
            },
            {
                "Name": "E",  # Young's modulus
                "Type": "LogNormal",
                "Moments": [3e10, 4.5e9],  # (N/m^2)
            },
            {
                "Name": "p",  # constant distributed load
                "Type": "LogNormal",
                # "Moments": [12000, 600],  # (N/m)
                "Moments": [1e4, 2e3],  # (N/m)
            },
        ]
    }
    myPriorDist = uq.createInput(PriorOpts)

    V_mid = np.array([12.84, 13.12, 12.13, 12.19, 12.67]) / 1000  # (m)
    myData = {
        "y": V_mid.tolist(),
        "Name": "Beam mid-span deflection",
    }
    BayesOpts = {
        "Type": "Inversion",
        "Data": myData,
        "Solver": {
            "Type": "MCMC",
            "MCMC": {
                "Steps": 20,
                "NChains": 100
            }
        }        
    }
    myBayesianAnalysis = uq.createAnalysis(BayesOpts)
    pointEstimate = [[1.2,	1.4,	131.5,	36910105055,	8620,	0.00010131]]

    assert np.array(myBayesianAnalysis['Results']['Sample']).shape == (BayesOpts["Solver"]["MCMC"]["Steps"],6,BayesOpts["Solver"]["MCMC"]["NChains"])

    burnIn = 0.5
    badChains = [1,2,3]
    prior = 800
    priorPredictive = 900
    posteriorPredictive = 1000

    myBayesianAnalysis = uq.postProcessInversion(myBayesianAnalysis,
                        'badChains', badChains, 
                        'pointEstimate', pointEstimate, 
                        'burnIn', burnIn,  
                        'percentiles', [.025, .975], 
                        'dependence', 1, 
                        'prior', prior, 
                        'priorPredictive', priorPredictive, 
                        'posteriorPredictive', posteriorPredictive, 
                        'gelmanRubin', 1,                        
                    )

    assert np.array(myBayesianAnalysis["Results"]["PostProc"]["PostSample"]).shape == (int(burnIn*BayesOpts["Solver"]["MCMC"]["Steps"]),
                                                                                    6,
                                                                                    BayesOpts["Solver"]["MCMC"]["NChains"]-len(badChains))
    assert (np.array(myBayesianAnalysis["Results"]["PostProc"]["PointEstimate"]["X"]) == np.array(pointEstimate)).all()
    assert myBayesianAnalysis["Results"]["PostProc"]["PointEstimate"]["Type"] == "custom"
    assert np.array(myBayesianAnalysis["Results"]["PostProc"]["PriorSample"]).shape == (prior, 6)
    assert np.array(myBayesianAnalysis["Results"]["PostProc"]["PriorPredSample"]["Sample"],ndmin=2).shape == (1, priorPredictive)
    assert np.array(myBayesianAnalysis["Results"]["PostProc"]["PostPredSample"]["Sample"],ndmin=2).shape == (1, posteriorPredictive)
    mySession.quit()
    time.sleep(1)    


