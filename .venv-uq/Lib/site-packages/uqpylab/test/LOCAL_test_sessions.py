import pytest
from uqpylab import sessions
#from uqlab_standalone import sessions as uq_session


SHost = "http://localhost:8000"
SToken = "636fd25c8c0aa05e344c0cc66abdcfb8c3af73d8"

mySession = sessions.cloud(host=SHost, token=SToken)
# (Optional) Get a convenient handle to the command line interface
uq = mySession.cli
# Reset the session
mySession.reset()

session_name = 'selftest-session'
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
print("Saving the session and quitting...")
mySession.save()
mySession.quit()
print("OK now starting a new session...")
mySession = sessions.cloud(host=SHost, token=SToken,force_restart=True)

uq = mySession.cli
print("Making sure that the input is not there...")
try: 
    uq.selectInput(InputName)
    assert False, "Test error. The input should not exist!"
except:
    pass
print("Loading the saved session...")
mySession.load(session_name)
print("Ensuring that the input is there now...")
InputToCheck = uq.getInput(InputName)
assert InputToCheck['Name'] == InputName
print("Current sessions:")
mySession.list()
print("Deleting the saved session...")
mySession.remove(session_name)
print("Current sessions:")
mySession.list()
    
print("Starting a session with force_restart flag on...")
mySession = sessions.cloud(host=SHost,token=SToken,force_restart=True)
uq = mySession.cli
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
mySession.hard_reset()
print("Creating an input...")
myInput = uq.createInput(InputOpts)
print("Done.")

