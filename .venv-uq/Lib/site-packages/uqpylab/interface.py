from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import uqpylab.helpers as helpers
import numpy as np
import pdb
import json

FORMAT_SPECS = {
    'create_module': {
        'request': 'JSON',
        'response': 'MAT'
    },
    'eval_model' : {
        'request': 'MAT',
        'response': 'MAT'
    },
    'input_function': {
        'request': 'JSON',
        'response': 'MAT'
    },
    'cli': {
        'request': 'JSON',
        'response': 'JSON'
    },
    'auxiliary': {
        'request': 'JSON',
        'response': 'JSON'
    },
    'auxiliary_binary_resp': {
        'request': 'JSON',
        'response': 'MAT'
    }
}


class uq():
    def __init__(self,session):
        self.session = session
        self.logger = session.logger
    def get_format_specs(self, method_type=None, **kwargs):
        if 'request_format' in kwargs:
            req_fmt = kwargs['request_format']
        else:
            req_fmt = FORMAT_SPECS[method_type.lower()]['request']

        if 'response_format' in kwargs:
            resp_fmt = kwargs['response_format']
        else:
            resp_fmt = FORMAT_SPECS[method_type.lower()]['response']

        return (req_fmt, resp_fmt)
    def uq_call(self, command, nargout=1, *args, **kwargs):
        REQ = {
        'Command' : command,
        'Argument' : [],
        'nargout' : nargout
        }
        for arg in args:
            if arg is not None:
                REQ["Argument"].append({'Value': arg})

        resp = self.session.rest_call(request=REQ, **kwargs)
        return self.session.get_resp_value(resp)


    def createModule(self, type, Opts, nargout=1, **kwargs):
        def _initialize_discrepancy_fields(discrepancy):
            TYPE_SPECIFIED = "Type" in discrepancy and discrepancy["Type"]
            PARAMS_SPECIFIED = "Parameters" in discrepancy and discrepancy["Parameters"]
            PRIOR_SPECIFIED = "Prior" in discrepancy and discrepancy["Prior"]
            self.session.logger.debug(f"Discrepancy: Type specified: {TYPE_SPECIFIED}, Parameters specified: {PARAMS_SPECIFIED}, Prior specified: {PRIOR_SPECIFIED}") 
            if not TYPE_SPECIFIED:
                discrepancy["Type"] = []                   
            if not PARAMS_SPECIFIED:
                discrepancy["Parameters"] = []
            if not PRIOR_SPECIFIED:
                discrepancy["Prior"] = []
            return discrepancy
        if 'Type' in Opts and Opts['Type']=='Inversion' and 'Discrepancy' in Opts:
            discrepancy = Opts["Discrepancy"]
            if isinstance(discrepancy, dict):
                Opts["Discprepancy"] = _initialize_discrepancy_fields(Opts["Discrepancy"])
            if isinstance(discrepancy, list):
                for idx, c_i in enumerate(discrepancy):
                    Opts["Discrepancy"][idx] = _initialize_discrepancy_fields(c_i)                
        REQ = {
            'Command' : 'uq_create{}'.format(type.lower().title()),
            'Argument' : [{'Value':Opts}],
            'nargout' : nargout,
        }
        (req_fmt, resp_fmt) = self.get_format_specs('create_module', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return self.session.get_resp_value(resp)
        # return True
    def createInput(self, InputOpts, **kwargs):
        def _initialize_copula_fields(copula):
            TYPE_SPECIFIED = "Type" in copula and copula["Type"]
            VARS_SPECIFIED = "Variables" in copula and copula["Variables"]  
            PARAMS_SPECIFIED = "Parameters" in copula and copula["Parameters"]
            RANKCORR_SPECIFIED = "RankCorr" in copula and copula["RankCorr"]  
            FAMILY_SPECIFIED = "Family" in copula and copula["Family"] 
            self.session.logger.debug("Copula: Type specified: {}, Variables specified: {}, Parameters specified: {}, RankCorr specified: {}, Family specified: {}".format(
            TYPE_SPECIFIED, VARS_SPECIFIED, PARAMS_SPECIFIED, RANKCORR_SPECIFIED, FAMILY_SPECIFIED)) 
            if not TYPE_SPECIFIED:
                msg = "You have to specify the type of the copula!"
                raise RuntimeError(msg)                    
            if not PARAMS_SPECIFIED:
                copula["Parameters"] = []
            if not RANKCORR_SPECIFIED:
                copula["RankCorr"] = []
            if not FAMILY_SPECIFIED:
                copula["Family"] = []
            return copula
        self.session.logger.debug("Starting the processing of Input options.")
        if "Marginals" in InputOpts:
            # Make sure that InputOpts["Marginals"] is a list
            if isinstance(InputOpts["Marginals"], dict):
                InputOpts["Marginals"] = [InputOpts["Marginals"]]

            for idx, marginal in enumerate(InputOpts["Marginals"]):
                PARAMS_SPECIFIED = "Parameters" in marginal and  marginal["Parameters"]
                MOMENTS_SPECIFIED = "Moments" in marginal and  marginal["Moments"]
                BOUNDS_SPECIFIED = "Bounds" in marginal and  marginal["Bounds"]
                INFERENCE_SPECIFIED = "Inference" in marginal and  marginal["Inference"]
                OPTIONS_SPECIFIED = "Options" in marginal and  marginal["Options"]
                self.session.logger.debug("Marginal nr. {}: Parameters specified: {}, Moments specified: {}".format(
                idx, PARAMS_SPECIFIED, MOMENTS_SPECIFIED))
                # Address the issue of mixed Moments-Parameters specification:
                # The fix here is to give the non-specified one as empty list
                if PARAMS_SPECIFIED and MOMENTS_SPECIFIED:
                    msg = "Cannot have both parameters and moments specified for a Marginal!"
                    raise RuntimeError(msg)
                if PARAMS_SPECIFIED:
                    # make sure that Parameters are like this: [[a,b]] and not
                    # like this: [a,b]
                    if not isinstance(marginal["Parameters"][0], list):
                        marginal["Parameters"] = [marginal["Parameters"]]
                    marginal["Moments"] = []
                if MOMENTS_SPECIFIED:
                    # make sure that Moments are like this: [[a,b]] and not
                    # like this: [a,b]
                    if not isinstance(marginal["Moments"][0], list):
                        marginal["Moments"] = [marginal["Moments"]]
                    marginal["Parameters"] = []
                if not BOUNDS_SPECIFIED:
                    marginal["Bounds"] = []
                if not (PARAMS_SPECIFIED or MOMENTS_SPECIFIED):
                    # This special case may happen e.g. in Kernel Smoothing type
                    marginal["Parameters"] = []
                    marginal["Moments"] = []
                if not INFERENCE_SPECIFIED:
                    marginal["Inference"] = []
                if not OPTIONS_SPECIFIED:
                    marginal["Options"] = []
        if "Copula" in InputOpts:
            copula = InputOpts["Copula"]
            if isinstance(copula, dict):
                InputOpts["Copula"] = _initialize_copula_fields(InputOpts["Copula"])
            if isinstance(copula, list):
                for idx, c_i in enumerate(copula):
                    InputOpts["Copula"][idx] = _initialize_copula_fields(c_i)
                 
        self.logger.debug(f"Updated InputOpts: {json.dumps(InputOpts)}")
        return self.createModule('input', InputOpts, **kwargs)
    def createModel(self, ModelOpts, **kwargs):
        default_comm_format_remote_models = 'JSON'
        # check the type of model:
        # Model: exists locally UNLESS it is an mString
        # MetaModel: exists remotely
        if ModelOpts['Type'].lower() in ['uq_default_model','model'] and 'mString' not in ModelOpts:
            # assign a compatible value to Type
            ModelOpts['Type'] = 'uq_default_model'
            # local definition
            if 'CommFormat' not in ModelOpts:
                ModelOpts['CommFormat'] = default_comm_format_remote_models
                req_fmt = 'JSON'
                resp_fmt = 'JSON'
            if ModelOpts['CommFormat'].lower() == 'json':
                ModelOpts['mFile'] = 'uq_cloud_remote'
                req_fmt = 'MAT'
                resp_fmt = 'MAT'
            elif ModelOpts['CommFormat'].lower() == 'mat':
                ModelOpts['mFile'] = 'uq_cloud_remote_mat'
            else:
                raise ValueError(f"Unknown CommFormat = {ModelOpts['CommFormat']}")
            # In this case the 'ModelFun' needs to be supplied to carry out model evaluations
            assert 'ModelFun' in ModelOpts, 'The ModelFun field is required and has not been supplied!'
            if 'Parameters' in ModelOpts:
                ModelOpts['Parameters']['theModel'] = ModelOpts['ModelFun']
            else:
                ModelOpts['Parameters'] = {'theModel': ModelOpts['ModelFun']}
            return self.createModule('model', ModelOpts)
        else:
            if ModelOpts['Type'].lower() in ['uq_default_model','model']:
                ModelOpts['Type'] = 'uq_default_model'
            # remote definition
            return self.createModule('model', ModelOpts, **kwargs)
    def createAnalysis(self, AnalOpts, **kwargs):
        return self.createModule('analysis', AnalOpts, **kwargs)
    def getSample(self, Input=None, N=None, Method='MC', Name=None, Value=None, nargout=1, **kwargs):

        REQ = {
                'Command' : 'uq_getSample',
                'nargout': nargout
            }
        if Input is None:
            REQ['Argument'] = [{'Value':N},{'Value':Method}]
        else:
            if type(Input) == str:
                REQ['Argument'] = [{'Value':Input},{'Value':N},{'Value':Method}]
            elif type(Input) == dict:
                REQ['Argument'] = [{'Value':Input['Name']},{'Value':N},{'Value':Method}]
            else:
                raise ValueError('Unsupported type of Input: {}!'.format(str(type(Input))))
        if Name is not None:
            REQ['Argument'] += [{'Value':Name}]
        if Value is not None:
            REQ['Argument'] += [{'Value':Value}]

        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)

        if nargout <=1:
            theSamples = np.array(resp_value, ndmin=2)
            # In case theSamples is a row vector we have to make a consistency check:
            # Is this correct (i.e. M>1 and N=1) or should it be transposed? (M=1 N>1)
            # For this we need some info from Input
            if theSamples.shape[0] == 1 and nargout == 1:
                if Input is None:
                    # in this case we really need the input so we have to make one more call
                    # to uqpylab
                    Input = self.getInput('')

                if type(Input['Marginals']) == dict:
                    M = 1
                else:
                    M = sum(type(x.get('Type')) == str for x in Input['Marginals'])
                if M == 1:
                    return theSamples.T
                else:
                    return theSamples
            else:
                return theSamples
        else:
            for idx, respi in enumerate(resp_value):
                respi = np.array(respi, ndmin=2)
                if N>1 and respi.shape[0] == 1:
                    resp_value[idx] = respi.T
                else:
                    resp_value[idx] = respi
            return resp_value

    def evalModel(self, Model=None, X=None, nargout=1, **kwargs):
        if X is None:
            return None
        # make sure that X is correctly interpreted when a single sample is provided
        X = np.array(X,ndmin=2,dtype=float)
        if Model is None:
            # If no Model is given get the currently selected one within the UQ session
            Model = self.getModel('')
        # Check the type of model
        isLOCAL = Model['Type'].lower() in ['model','uq_default_model']
        isLOCAL = isLOCAL and 'mString' not in Model
        if not isLOCAL:
            REQ = {
                'Command' : 'uq_evalModel',
                'Argument' : [{'Value': Model['Name']},
                              {'Value': helpers.jsonify_np_array(X)}],
                'nargout' : nargout
            }

            # if we reached this far the model exists on the uq_worker
            (req_fmt, resp_fmt) = self.get_format_specs('eval_model', **kwargs)
            resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
            resp_value = self.session.get_resp_value(resp)
        else:
            # Check whether there are parameters involved
            PARAMS_EXIST = len(Model['Parameters'])>1
            # Is the model vectorized?
            IS_VECTORIZED = Model['isVectorized']
            # Assume that 'ModelFun' field exists, but make sure it is callable
            if PARAMS_EXIST and IS_VECTORIZED:
                resp_value = helpers.function_eval(Model['ModelFun'], X, Model['Parameters'])
            elif PARAMS_EXIST and not IS_VECTORIZED:
                resp_value = []
                for x in X:
                    resp_value.append(helpers.function_eval(Model['ModelFun'], x, Model['Parameters']))
                resp_value = np.array(resp_value)
            elif not PARAMS_EXIST and IS_VECTORIZED:
                resp_value = helpers.function_eval(Model['ModelFun'], X)
            else: # not PARAMS_EXIST and not IS_VECTORIZED
                resp_value = []
                for x in X:
                    resp_value.append(helpers.function_eval(Model['ModelFun'], x))
                resp_value = np.array(resp_value)

        # we have to make a consistency check
        if nargout >1:
            for idx, respi in enumerate(resp_value):
                respi = np.array(respi, ndmin=2)
                if X.shape[0]>1 and respi.shape[0] == 1:
                    resp_value[idx] = respi.T
                else:
                    resp_value[idx] = respi
        if nargout == 1:
            resp_value = np.array(resp_value, ndmin=2)
            if X.shape[0]>1 and resp_value.shape[0] == 1:
                resp_value = resp_value.T
        return resp_value

    def getModule(self, thistype, name, **kwargs):
        if thistype == 'input':
            cmd = 'uq_getInput'
            if type(name)==dict:
                name = name["Name"]
        if thistype == 'model':
            cmd = 'uq_getModel'
            if type(name)==dict:
                name = name["Name"]            
        if thistype == 'analysis':
            cmd = 'uq_getAnalysis'
            if type(name)==dict:
                name = name["Name"]            
        REQ = {
            'Command' : cmd,
            'Argument' : [{'Value':name}]
        }
        (req_fmt, resp_fmt) = self.get_format_specs('cli', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return resp_value
    def getInput(self, name):
        return self.getModule('input', name)
    def getModel(self, name):
        return self.getModule('model', name)
    def getAnalysis(self, name):
        return self.getModule('analysis', name)
    def getAnalysisResults(self, name, **kwargs):
        REQ = {
            'Command' : 'uq_getAnalysisResults',
            'Argument' : [{'Value':name}]
        }
        
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary_binary_resp', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return resp_value
    def listModules(self, type, **kwargs):
        REQ = {
            'Command' : 'uq_listModules',
            'Argument' : [{'Value':type}],
            'nargout'  : 3
        }
        (req_fmt, resp_fmt) = self.get_format_specs('cli', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        print(resp[2]['Value'])
    def listInputs(self):
        self.listModules('input')
    def listModels(self):
        self.listModules('model')
    def listAnalyses(self):
        self.listModules('analysis')
    def selectModule(self, thistype, name, **kwargs):
        if thistype == 'input':
            cmd = 'uq_selectInput'
            if type(name)==dict:
               name = name["Name"]
        if thistype == 'model':
            cmd = 'uq_selectModel'
            if type(name)==dict:
               name = name["Name"]            
        if thistype == 'analysis':
            cmd = 'uq_selectAnalysis'
            if type(name)==dict:
               name = name["Name"]            

        REQ = {
            'Command' : cmd,
            'Argument' : [{'Value':name}]
        }
        (req_fmt, resp_fmt) = self.get_format_specs('cli', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return 'OK'
    def selectInput(self,name):
        self.selectModule('input', name)
    def selectModel(self,name):
        self.selectModule('model', name)
    def selectAnalysis(self,name):
        self.selectModule('analysis', name)
    def removeModule(self, name, module_type, **kwargs):
        REQ = {
            'Command' : 'uq_removeModule',
            'Argument' : [{'Value':name},{'Value':module_type.lower()}]
        }
        (req_fmt, resp_fmt) = self.get_format_specs('cli', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return 'OK'
    def print(self, obj=None, name=None, type=None, **kwargs):
        if obj is None:
            moduleToPrint = {
                "Name": name,
                "Class": type
            }
        else:
            moduleToPrint = {
                "Name": obj["Name"],
                "Class": obj["Class"]
            }
        REQ = {
            'Command' : 'uq_print',
            'Argument' : [{'Value':moduleToPrint}]
        }
        (req_fmt, resp_fmt) = self.get_format_specs('cli', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        print(resp_value)

    def display(self, obj=None, name=None, type=None, show=True, **kwargs):
        fig = self.make_fig(obj, name, type, **kwargs)
        if 'test_mode' in kwargs and kwargs['test_mode']:
            return fig
        if show:
            self.show_fig(fig)
        return fig

    def make_fig(self, obj=None, name=None, type=None, **kwargs):
        if obj is None:
            obj = self.getModule(type=type, name=name)
        return helpers.display(obj, self, **kwargs)

    def show_fig(self, fig):
        if type(fig) == list:
            for i in range(len(fig)):
                fig[i].show()
        elif fig is None:
            print("None is not printed. Something went wrong or the display routine still has not been implemented yet.")
        else:
            fig.show()

    def rng(self, seed=None, generator=None, **kwargs):
        if seed is None and generator is None:
            REQ = {
                'Command' : 'uq_rng',
            }
        elif generator is None:
            # No generator specified but the seed is not None
            REQ = {
            'Command' : 'uq_rng',
            'Argument' : [{'Value':seed}]
            }
        elif seed is None:
            # No seed specified but the generator is not None: return Error
            raise ValueError('The seed value is required when specifying the generator.')
        else:
            # Both seed and generator specified
            REQ = {
                'Command' : 'uq_rng',
                'Argument' : [{'Value':seed},{'Value':generator}]
            }
        (req_fmt, resp_fmt) = self.get_format_specs('cli', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return resp_value

    def all_f(self,function=None, X=None, Marginals=None, **kwargs):
        if X is None or Marginals is None:
            return None
        REQ = {
            'Command' : function,
            'Argument' : [{'Value': helpers.jsonify_np_array(X)},
                          {'Value': Marginals}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape(X.shape)
    def all_cdf(self, X=None, Marginals=None, **kwargs):
        return self.all_f(function='uq_all_cdf', X=X, Marginals=Marginals, **kwargs)
    def all_pdf(self, X=None, Marginals=None, **kwargs):
        return self.all_f(function='uq_all_pdf', X=X, Marginals=Marginals, **kwargs)
    def all_invcdf(self, F=None, Marginals=None, **kwargs):
        return self.all_f(function='uq_all_invcdf', X=F, Marginals=Marginals, **kwargs)

    def GeneralIsopTransform(self, X, X_Marginals, X_Copula, Y_Marginals, Y_Copula, **kwargs):
        REQ = {
            'Command' : 'uq_GeneralIsopTransform',
            'Argument' : [{'Value': helpers.jsonify_np_array(X)},
                          {'Value': X_Marginals},
                          {'Value': X_Copula},
                          {'Value': Y_Marginals},
                          {'Value': Y_Copula}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape(X.shape)
    def IsopTransform(self, X, X_marginals, Y_marginals, **kwargs):
        REQ = {
            'Command' : 'uq_IsopTransform',
            'Argument': [{'Value': helpers.jsonify_np_array(X)},
                         {'Value': X_marginals},
                         {'Value': Y_marginals}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape(X.shape) 
    def NatafTransform(self, X, marginals, copula, **kwargs):
        REQ = {
            'Command' : 'uq_NatafTransform',
            'Argument': [{'Value': helpers.jsonify_np_array(X)},
                         {'Value': marginals},
                         {'Value': copula}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape(X.shape) 
    def invNatafTransform(self, Uin, marginals, copula, **kwargs):
        REQ = {
            'Command' : 'uq_invNatafTransform',
            'Argument': [{'Value': helpers.jsonify_np_array(Uin)},
                         {'Value': marginals},
                         {'Value': copula}],
            'nargout' : 1
        }        
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape(Uin.shape) 
    def RosenblattTransform(self, X, marginals, copula, **kwargs):
        REQ = {
            'Command' : 'uq_RosenblattTransform',
            'Argument': [{'Value': helpers.jsonify_np_array(X)},
                         {'Value': marginals},
                         {'Value': copula}],
            'nargout' : 1
        }        
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape(X.shape) 
    def invRosenblattTransform(self, Z, marginals, copula, **kwargs):
        REQ = {
            'Command' : 'uq_invRosenblattTransform',
            'Argument': [{'Value': helpers.jsonify_np_array(Z)},
                         {'Value': marginals},
                         {'Value': copula}],
            'nargout' : 1
        }        
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape(Z.shape) 

    def extractObj(self, parentName, objPath, objType, **kwargs):
        REQ = {
            'Command' : 'uq_extractObj',
            'Argument' : [{'Value': parentName},
                          {'Value': objPath},
                          {'Value': objType}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary_binary_resp', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return resp_value
    def extractFromInput(self, parentName, objPath, **kwargs):
        return self.extractObj(parentName=parentName, objPath=objPath, objType="input", **kwargs)
    def extractFromModel(self, parentName, objPath, **kwargs):
        return self.extractObj(parentName=parentName, objPath=objPath, objType="model", **kwargs)
    def extractFromAnalysis(self, parentName, objPath, **kwargs):
        return self.extractObj(parentName=parentName, objPath=objPath, objType="analysis", **kwargs)

    def eval_Kernel(self, X1, X2, theta, Options, **kwargs):
        theta = np.asarray(theta)
        REQ = {
            'Command' : 'uq_eval_Kernel',
            'Argument' : [{'Value': helpers.jsonify_np_array(X1)},
                          {'Value': helpers.jsonify_np_array(X2)},
                          {'Value':  helpers.jsonify_np_array(theta)},
                          {'Value':  Options}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary_binary_resp', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return np.asarray(resp_value)

    # Input module, Copula-specific functions
    def PairCopula(self, family, theta, rotation=None, **kwargs):
        REQ = {
            'Command' : 'uq_PairCopula',
            'Argument' : [{'Value': family},
                          {'Value': theta}],
            'nargout' : 1
        }
        if rotation is not None:
            REQ['Argument'].append({'Value': rotation})

        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary_binary_resp', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return self.session.get_resp_value(resp)


    def PairCopulaOperation(self, copula, operation, *args, **kwargs):
        REQ = {
        'Command' : operation,
        'Argument' : [{'Value': copula}],
        'nargout' : 1
        }
        for arg in args:
            if arg is not None:
                REQ["Argument"].append({'Value': arg})

        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary_binary_resp', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return self.session.get_resp_value(resp)

    def PairCopulaKendallTau(self, copula, **kwargs):
        return self.PairCopulaOperation(copula, 'uq_PairCopulaKendallTau', **kwargs)
    def PairCopulaUpperTailDep(self, copula, **kwargs):
        return self.PairCopulaOperation(copula, 'uq_PairCopulaUpperTailDep', **kwargs)
    def PairCopulaLowerTailDep(self, copula, **kwargs):
        return self.PairCopulaOperation(copula, 'uq_PairCopulaLowerTailDep', **kwargs)
    def CopulaSummary(self, copula, *args, **kwargs):
        message = self.PairCopulaOperation(copula, 'uq_CopulaSummary', response_format='JSON', *args, **kwargs)
        if 'no_print' in kwargs and kwargs['no_print']:
            return message
        else:
            print(message)
        

    def test_block_independence(self, X, alpha, stat='Kendall', correction='auto', verbose=1, nargout=4, **kwargs):
        if isinstance(X, np.ndarray):
            X = X.tolist()
        assert type(X) is list, "Requirement not satisfied: X should be a list!"
        REQ = {
            'Command' : 'uq_test_block_independence',
            'Argument' : [{'Value': X},
                          {'Value': alpha},
                          {'Value': stat},
                          {'Value': correction},
                          {'Value': verbose}],
            'nargout' : nargout
        }
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return self.session.get_resp_value(resp)

    def VineCopula(self, cop_type, structure, families, parameters, *args, **kwargs):
        return self.uq_call('uq_VineCopula', 1,
            cop_type, structure, families, parameters, *args)
    def GaussianCopula(self, C, corrType='Linear', **kwargs):
        if isinstance(C, np.ndarray):
            C = C.tolist()
        return self.uq_call('uq_GaussianCopula', 1, C, corrType)
    def IndepCopula(self, M, **kwargs):
        return self.PairCopulaOperation('uq_IndepCopula',1, M, **kwargs)

    def Marginals(self, M, Type, Parameters=None, Moments=None, flatten=True):
        if Moments is not None and Parameters is not None:
            msg = 'Cannot have both parameters and moments specified for a Marginal!'
            raise ValueError(msg)
        if Moments is None:
            Marginals =  [{'Type':Type, 'Parameters': Parameters} for i in range(M)]
        elif Parameters is None:
            Marginals =  [{'Type':Type, 'Moments': Moments} for i in range(M)]
        else:
            # This special case may happen e.g. in Kernel Smoothing type
            Marginals =  [{'Type':Type, 'Parameters': [], 'Moments': []} for i in range(M)]
        
        if flatten and len(Marginals) == 1:
            # If the flatten flag is set to True and there is only one Marginal specified
            # return the single Marginal as a dictionary instead of wrapping it inside a list
            return Marginals[0]
        else:
            return Marginals
    
    def StdNormalMarginals(self, D, **kwargs):
        Type = 'Gaussian'
        Parameters = [0, 1]
        Marginals =  [{'Type':Type, 'Parameters': Parameters} for i in range(D)]
        return Marginals

    def StdUniformMarginals(self, M, **kwargs):
        Type = 'Uniform'
        Parameters = [0, 1]
        Marginals =  [{'Type':Type, 'Parameters': Parameters} for i in range(M)]
        return Marginals

    def setDefaultSampling(self, Input=None, Method=None, **kwargs):
        if Input is None:
            current_input = self.getInput('')
        if type(Input) == dict:
            current_input = self.getInput(Input["Name"])
        elif type(Input) == str:
            current_input = self.getInput(Input)
        REQ = {
            'Command' : 'uq_setDefaultSampling',
            'Argument': [{'Value': current_input["Name"]},
                         {'Value': Method}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return self.session.get_resp_value(resp)

    def enrichDoE(self, function, X0=None, N=None, Input=None, nargout=1, **kwargs):
        if Input is None:
            Input = self.getInput('')        
        REQ = {
            'Command' : function,
            'Argument': [{'Value': X0.tolist()},
                         {'Value': N},
                         {'Value': Input["Name"]}],
            'nargout' : nargout
        }
        return self.variOutVars(REQ=REQ, N=N, nargout=nargout, **kwargs)
    def enrichLHS(self, X0=None, N=None, Input=None, nargout=1, **kwargs):
        return self.enrichDoE('uq_enrichLHS', X0, N, Input, nargout, **kwargs)
    def LHSify(self, X0=None, N=None, Input=None, nargout=1, **kwargs):
        return self.enrichDoE('uq_LHSify', X0, N, Input, nargout, **kwargs)    
    def enrichSobol(self, X0=None, N=None, Input=None, nargout=1, **kwargs):
        return self.enrichDoE('uq_enrichSobol', X0, N, Input, nargout, **kwargs)    
    def enrichHalton(self, X0=None, N=None, Input=None, nargout=1, **kwargs):
        return self.enrichDoE('uq_enrichHalton', X0, N, Input, nargout, **kwargs)        

    def KernelMarginals(self, X, bounds=[], bandwidth=[], **kwargs):
        # X = np.asarray(X)
        setBounds = (1 if (type(bounds)==list and bounds) or 
                          type(bounds) == int or
                          type(bounds) == float 
                          else 0)
        setBandwidth = (1 if (type(bandwidth)==list and bandwidth) or 
                              type(bandwidth) == int or
                              type(bandwidth) == float 
                          else 0)
        M = X.shape[1]

        ## Build an array Mx2 with the bounds of each variable, if specified
        if (type(bounds) == int or type(bounds) == float) and bounds >= 0: 
            # bounds is a positive scalar
            maxX = np.amax(X, axis=0)
            minX = np.amin(X, axis=0)
            deltaX = maxX - minX
            minData = minX-bounds*deltaX
            maxData = maxX+bounds*deltaX
            Bounds = np.stack((minData, maxData),axis=-1).tolist()
        elif len(bounds) == M and all([(len(i) == 2) or (len(i) == 0) for i in bounds]):
            # bounds is a regular array OR mimicks the cell array (with possible empty elements)
            Bounds = bounds
        elif len(bounds) == 0:
            pass
        else:
            msg = 'Bounds must be a positive number, or an array Mx2!'
            raise RuntimeError(msg)
        
        ## Build an array Mx1 with the kernel bandwidth for each variable, if wanted
        if (type(bandwidth) == int or type(bandwidth) == float) and bandwidth >= 0:
            Bandwidth = (bandwidth * np.ones((M, 1))).tolist()
        elif len(bandwidth) == M:
            Bandwidth = bandwidth
        elif len(bandwidth) == 0:
            pass
        else:
            msg = 'input variable bandwidth must be a positive number or an array of M possitive numbers'
            raise RuntimeError(msg)
        Marginals = []
        for jj in range(M):
            var = {
                'Type': 'ks',
                'Parameters': X[:,jj].tolist(),
                'Options': {'Kernel': 'Normal'}
            }
            if setBounds:
                if len(Bounds[jj])==0:
                    maxX = np.amax(X[:,jj], axis=0)
                    minX = np.amin(X[:,jj], axis=0)
                    dX=1e2*(maxX-minX)
                    Bounds[jj] = [minX-dX, maxX+dX]
                elif len(Bounds[jj])!=2:
                    msg = f'uq_KernelInput: bounds[{jj}] must be a 1x2 array'
                    raise RuntimeError(msg)   
                var['Bounds'] = Bounds[jj]
            if setBandwidth:
                var['Options']['Bandwidth'] = Bandwidth[jj]
            Marginals.append(var)
        return Marginals
   
    def PairCopulaParameterRange(self, family=None, k='all'):
        REQ = {
            'Command' : 'uq_PairCopulaParameterRange',
            'Argument': [{'Value': family},
                         {'Value': k}],
            'nargout' : 1
        }
        resp = self.session.rest_call(request=REQ)
        resp_value = self.session.get_resp_value(resp)
        if resp_value and resp_value[-1] is None:
                resp_value[-1] = float('inf')
        return resp_value
    
    def variOutVars(self, REQ, N, nargout, **kwargs):
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        if nargout <=1:
            theSamples = np.array(resp_value, ndmin=2)
            # In case theSamples is a row vector we have to make a consistency check:
            # Is this correct (i.e. M>1 and N=1) or should it be transposed? (M=1 N>1)
            # For this we need some info from Input
            if theSamples.shape[0] == 1 and nargout == 1:
                if Input is None:
                    # in this case we really need the input so we have to make one more call
                    # to uqpylab
                    Input = self.getInput('')

                if type(Input['Marginals']) == dict:
                    M = 1
                else:
                    M = sum(type(x.get('Type')) == str for x in Input['Marginals'])
                if M == 1:
                    return theSamples.T
                else:
                    return theSamples
            else:
                return theSamples
        else:
            for idx, respi in enumerate(resp_value):
                respi = np.array(respi, ndmin=2)
                if N>1 and respi.shape[0] == 1:
                    resp_value[idx] = respi.T
                else:
                    resp_value[idx] = respi
            return resp_value

    def subsample(self, X=None, NK=None, Method=None, Name=None, Value=None, nargout=1, **kwargs):
        REQ = {
            'Command' : 'uq_subsample',
            'Argument': [{'Value': X.tolist()},
                         {'Value': NK},
                         {'Value': Method},
                         {'Value': Name},
                         {'Value': Value}],
            'nargout' : nargout
        }
        return self.variOutVars(REQ=REQ, N=NK, nargout=nargout)

    def sampleU(self, N=None, M=None, options=None, **kwargs):
        REQ = {
            'Command' : 'uq_sampleU',
            'Argument': [{'Value': N},
                            {'Value': M}],
            'nargout' : 1
        }       
        if options is not None:
            REQ['Argument'].append({'Value': options})
        (req_fmt, resp_fmt) = self.get_format_specs('input_function', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = np.array(self.session.get_resp_value(resp), ndmin=2)
        return resp_value.reshape((N,M)) 

    def MarginalFields(self, marginals, **kwargs):
        REQ = {
            'Command' : 'uq_MarginalFields',
            'Argument': [{'Value': marginals}],
            'nargout' : 1
        }  
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return self.session.get_resp_value(resp)
    
    def estimateMoments(self, marginal, *args, nargout=1, **kwargs):
        REQ = {
            'Command' : 'uq_estimateMoments',
            'Argument': [{'Value': marginal}],
            'nargout' : nargout
        }
        if len(args) > 0:
            for i in range(0,len(args)):
                REQ['Argument'].append({'Value': args[i]})
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return tuple(self.session.get_resp_value(resp))

    def repeatAnalysis(self, obj, N=1, **kwargs):
        REQ = {
            'Command' : 'uq_repeatAnalysis',
            'Argument': [{'Value': obj['Name']},
                         {'Value': N}],
            'nargout' : 1
        }
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary', **kwargs)
        return self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)        
    
    def resumeAnalysis(self, *args, **kwargs):
        REQ = {
            'Command' : 'uq_resumeAnalysis',
            'Argument': [{'Value': args[0]}],
            'nargout' : 1
        }  
        if len(args) == 2:
            REQ['Argument'].append({'Value': args[1]})
        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        return self.session.get_resp_value(resp)
    
    def display_bar(self, data, **kwargs):
        return helpers.display_bar(data, **kwargs)
    
    def postProcessInversion(self, obj, *nargs, **kwargs):
        REQ = {
            'Command': 'uq_postProcessInversion',
            'Argument' : [{'Value': obj['Name']}],
            'nargout': 1
        }

        if len(nargs) > 0:
            for narg in nargs:
                REQ['Argument'].append({'Value': narg})

        (req_fmt, resp_fmt) = self.get_format_specs('auxiliary_binary_resp', **kwargs)
        resp = self.session.rest_call(request=REQ, request_format=req_fmt, response_format=resp_fmt)
        resp_value = self.session.get_resp_value(resp)
        return resp_value
