import pytest
from uqpylab import sessions
#from uqlab_standalone import sessions as uq_session
import time

# def start_a_session(host_spec, token_spec):
#     if host_spec and token_spec:
#         print(f"Connecting to UQCloud host: {host_spec}, with token: {token_spec}")
#         return sessions.cloud(host=host_spec,token=token_spec, log_level='DEBUG')
#     else:
#         print("Using remote UQCloud")
#         return sessions.cloud(log_level='DEBUG')

def test_save_load(request, helpers):
    print("Starting a session...")
    mySession = helpers.init_session(request)
    session_name = 'selftest-session'
    uq = mySession.cli
    uq.rng(100, 'twister')
    mySession.reset()
    mySession.rename(session_name)
    InputOpts = {
        'Marginals': [{
            'Type':'Gaussian',
            'Parameters': [0.10, 0.0161812]
            } ,
            {
            'Type':'Lognormal',
            'Parameters' : [7.71, 1.0056]
            },
            {
            'Type':'Uniform',
            'Parameters' : [63070, 115600]
            }]
        }
    print("Creating an input...")
    myInput = uq.createInput(InputOpts)
    InputName = myInput['Name']
    print("Saving the session and quiting...")
    mySession.save()
    mySession.quit()
    time.sleep(1)    
    print("OK now starting a new session...")
    mySession = helpers.init_session(request)
    uq = mySession.cli
    print("Making sure that the input is not there...")
    try: 
        uq.selectInput(InputName)
        assert False, "Test error. The input should not exist!"
    except:
        pass
    print("Loading the saved session...")
    mySession.load(session_name)
    time.sleep(1)    
    print("Ensuring that the input is there now...")
    InputToCheck = uq.getInput(InputName)
    assert InputToCheck['Name'] == InputName
    print("Current sessions:")
    mySession.list()
    print("Deleting the saved session...")
    mySession.remove(session_name)
    time.sleep(1)    
    print("Current sessions:")
    mySession.list()
    
def test_restarts(request, helpers):
    host_spec = request.config.getoption('--host')
    token_spec = request.config.getoption('--token')
    print("Starting a session with force_restart flag on...")
    mySession = sessions.cloud(host=host_spec,token=token_spec, force_restart=True)
    uq = mySession.cli
    time.sleep(1)    
    uq.rng(100, 'twister')
    # Let's do some random operation to make sure that the session is functional
    InputOpts = {
        'Marginals': [{
            'Type':'Gaussian',
            'Parameters': [0.10, 0.0161812]
            } ,
            {
            'Type':'Lognormal',
            'Parameters' : [7.71, 1.0056]
            },
            {
            'Type':'Uniform',
            'Parameters' : [63070, 115600]
            }]
        }
    print("Creating an input...")
    myInput = uq.createInput(InputOpts)
    print("Do another restart using the already initialised session object...")
    time.sleep(1)    
    mySession.hard_reset()
    time.sleep(1)    
    print("Creating an input...")
    myInput = uq.createInput(InputOpts)
    print("Done.")
    time.sleep(1)    
    mySession.quit()
    time.sleep(1)    
