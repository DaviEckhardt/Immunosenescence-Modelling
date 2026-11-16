import pytest
import uqpylab.sessions as uq_session

TESTS_LOG_LEVEL = 'INFO' # Recommended is 'INFO' or 'DEBUG' 

def pytest_addoption(parser):
    parser.addoption(
        '--local', action='store', default=False, help='Use localhost deployment of UQCloud?'
    )
    parser.addoption(
        '--host', action='store', default=None, help='Manually specify UQCloud host'
    )
    parser.addoption(
        '--token', action='store', default=None, help='Manually specify access token'
    )

class Helpers:
    """
    Helper functions that are used in unit tests
    """
    @staticmethod
    def init_session(request):
        host_spec = request.config.getoption('--host')
        token_spec = request.config.getoption('--token')
        if host_spec and token_spec:
            print(f"Connecting to UQCloud host: {host_spec}, with token: {token_spec}")
            mySession = uq_session.cloud(host=host_spec,token=token_spec, log_level=TESTS_LOG_LEVEL)
        else:
            print("Using remote UQCloud")
            mySession = uq_session.cloud(log_level=TESTS_LOG_LEVEL)
        mySession.reset()
        return mySession
    

@pytest.fixture
def helpers():
    return Helpers