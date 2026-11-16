import pdb
import scipy.io as spio
from importlib import import_module,util
import uqpylab.display_util as display_util
import uqpylab.display_sensitivity as display_sensitivity
import uqpylab.display_general as display_general
import uqpylab.display_reliability as display_reliability
import uqpylab.display_inversion as display_inversion
import numpy as np
from appdirs import user_data_dir
import os
from pathlib import Path
import json
# Helper functions
def uqcloud_save_config(host,token):
    # Saves the UQCloud url and token of the user to a file in a proper AppDir
    # if this specific file is found during import of the UQpyLab package
    # those values will be automatically loaded
    theDir = user_data_dir("UQCloud","UQCloud")
    # Create the folder if it doesn't exist
    Path(theDir).mkdir(parents=True, exist_ok=True)
    theFile = os.path.join(theDir, 'uqcloud_config.json')
    theConfig = {
    	"host": host,
    	"token": token
    }
    with open(theFile,'w') as outfile:
        json.dump(theConfig, outfile)
    return "Stored UQCloud configuration in {}".format(theFile)
def uqcloud_load_config(verbose=False):
    # Loads the UQCloud url and token from the configuration file
    theDir = user_data_dir("UQCloud","UQCloud")
    theFile = os.path.join(theDir, 'uqcloud_config.json')
    if os.path.isfile(theFile):
        with open(theFile,'r') as infile:
            theConfig = json.load(infile)
    else:
        theConfig = {
        	"host" : None,
        	"token": None
        }
    if verbose:
        print("Loaded UQCloud configuration from file {}.".format(theFile))
    return theConfig
def load_module(path, name):
    # Loads a module from a given path with a given name
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def jsonify_np_array(A):
    if isinstance(A, np.ndarray):
        return A.tolist()
    else:
        return A
def getlines(fd):
    line = bytearray()
    c = None
    while True:
        c = fd.read(1)
        if c is None:
            return
        line += c
        if c == b'\n':
            yield str(line)
            del line[:]
def extract_json(str):
    # extracts the json raw sub-string from a string
    idx_start_candidates = [str.find('{'), str.find('[')]
    json_start = min(x for x in idx_start_candidates if x != -1)
    idx_end_candidates = [str.rfind('}'), str.rfind(']')]
    json_end = max(idx_end_candidates) + 1
    return str[json_start : json_end]


def local_configuration():
    config = ConfigParser()
    #config.read('config.ini')
    section = 'workspace'
    ROOT_PATH = path.dirname(path.realpath(__file__))
    confpath = path.join(ROOT_PATH,'config.ini')
    config.add_section(section)
    print("Configuring for target platform: {}".format(sys.platform))
    if sys.platform.startswith('win'):
        config.set(section,'uqlab_dep', path.join(ROOT_PATH,'uq_web_cli_tcpbased.exe'))
        config.set(section,'uqlab_watchdog', path.join(ROOT_PATH,'uq_background_cleaner.exe'))
    else:
        config.set(section,'uqlab_dep', path.join(ROOT_PATH,'uq_web_cli_tcpbased'))
        config.set(section,'uqlab_watchdog', path.join(ROOT_PATH,'uq_background_cleaner'))
    print('Done.')
    MCR_LOCAL = input("Please provide the absolute path of your Matlab Runtime Installation: ")
    config.set(section, 'mcr_path', MCR_LOCAL)
    with open(confpath, 'w') as conffile:
        config.write(conffile)
    print("Local configuration:\n{}\nhas been succesfully updated.".format(confpath))

def function_eval(fun, X, Parameters=None):
    if callable(fun):
        return fun(X)
    else:
        # expecting the fun to be a string
        # we are expecting a syntax a.b.c (or longer) where a.b is the module and
        # c is the method
        X = np.array(X)
        method_sep = fun.rfind('.')
        the_module = import_module(fun[0:method_sep])
        the_method = getattr(the_module, fun[method_sep+1:])
        if Parameters is None:
            return the_method(X)
        else:
            return the_method(X, Parameters)

def display(obj, theInterface, **kwargs):
    if obj['Class'] == 'uq_input':
        # display input
        return display_util.Input(obj, theInterface, **kwargs)
    if obj['Class'] == 'uq_model':
        # display model
        if obj['MetaType'].lower() == 'pce':
            return display_util.PCE(obj, **kwargs)
        if obj['MetaType'].lower() == 'kriging':
            return display_util.Kriging(obj, theInterface, **kwargs)
        if obj['MetaType'].lower() == 'pck':
            return display_util.PCK(obj, theInterface, **kwargs)
        if  obj['MetaType'].lower() == 'lra':
            return display_util.LRA(obj, **kwargs)
        if  obj['MetaType'].lower() == 'svr':
            return display_util.SVR(obj, **kwargs)
        if  obj['MetaType'].lower() == 'svc':
            return display_util.SVC(obj, **kwargs)
    if obj['Class'] == 'uq_analysis':
        if obj['Type'] == 'uq_sensitivity':
            return display_sensitivity.sensitivity(obj, **kwargs)  
        elif obj['Type'] == 'uq_reliability':
            return display_reliability.reliability(obj, theInterface, **kwargs)
        elif obj['Type'] == 'uq_inversion':
            return display_inversion.inversion(obj, theInterface, **kwargs)
        else:
            # display analysis
            print("Not yet implemented!")

def display_bar(data, **kwargs):
    return display_general.display_bar(data, **kwargs)

def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], spio.matlab.mat_struct):
            d[key] = _todict(d[key])
    return d

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _tolist(elem.tolist())
        elif isinstance(elem, np.generic):
            d[strg] = elem.item()    
        else:
            d[strg] = elem
    return d

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    try:
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem.tolist()))
            elif isinstance(sub_elem, np.generic):
                elem_list.append(sub_elem.item())  
            else:
                elem_list.append(sub_elem)
        return elem_list
    except:
        pass

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def isnumeric(variable):
    '''
    this function checks if the variable or the elements within a list are numeric
    '''
    numeric_types = (int, float, complex, np.int_, np.float_, np.complex_)
    
    if isinstance(variable, numeric_types):
        return True
    
    if isinstance(variable, list):
        return all(isinstance(item, numeric_types) for item in variable)
    
def isListofDictionaries(variable):
    '''
    this function checks if the variable is a list of dictionaries
    '''
    if isinstance(variable, list):
        return all(isinstance(item, dict) for item in variable)
    return False

def hasNDimensionalStructure(lst, N, depth=0):
    '''
    this function checks whether a non-regular nested list has a structure that allows accessing elements with N indices
    '''
    if depth == N:
        return True
    if isinstance(lst, list):
        for item in lst:
            if hasNDimensionalStructure(item, N, depth + 1):
                return True
    return False
