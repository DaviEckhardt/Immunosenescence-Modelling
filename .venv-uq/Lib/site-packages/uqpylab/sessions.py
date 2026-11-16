import uqpylab.helpers as helpers
import uqpylab.interface as interface
import json
import ssl
import requests
import numpy as np
import time
import sys
import uuid
import tempfile
import io 
import logging
from scipy.io import savemat, loadmat

MAT_REQUEST_VARIABLE = 'REQ'
MAT_RESPONSE_VARIABLE = 'RESP'
VERSION = '1.0'

class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})): pass

class uq_session():
    def __init__(self, name=None, log_level=None,log_format=None):
        """
        Initializer of any session object. See also cloud.__init__ for more 
        initialization steps that are taken for cloud-specific sessions.
        """
        # Try to retrieve the UQCloud configuration from a file if available
        theConfig = helpers.uqcloud_load_config()
        self.stored_host = theConfig['host']
        self.stored_token = theConfig['token']
        self.version = VERSION
        if name is None:
            self.name = uuid.uuid4().hex
        else:
            self.name = name
        # Set the logging output format
        if log_format is None:
            # Set the default log format if not specified
            log_format = ' %(name)s :: %(levelname)-8s :: %(message)s'
        # Initialize a logger
        logging.basicConfig(stream = sys.stdout, format=log_format)
        self.logger = logging.getLogger(__name__)
        # Set the logger log level
        self.set_log_level(log_level=log_level)
        # Create an empty cache property (to be used as cache in some advanced use cases)
        self.cache = None 

    def set_log_level(self, log_level=None):
        """
        Helper function for assigning a log level (INFO, WARN, ERROR, etc.) if specified. 
        """
        if log_level is not None:
            level = logging.getLevelName(log_level.upper())
            self.logger.setLevel(level)
    def rename(self, new_name):
        """
        Helper function for renaming a session. 
        """
        name_prev = self.name
        self.name = new_name
        return "Session renamed from {} to {}.".format(name_prev, new_name)
    def throw_worker_error(self,msg):
        """
        Helper function for throwing errors that were encountered by the UQLab worker
        during the execution of some job.
        """
        raise RuntimeError(msg)
    def get_resp_value(self, resp):
        """
        Unpacks the values out of a response that has been received from UQCloud.
        If the request resulted in some error during the execution by UQCloud,
        the error is rethrown locally.  
        """
        # Build a response list if resp contains more than 1 'Value' keys
        if sum([1 for d in resp if 'Value' in d]) > 1:
            resp_final=[]
            for respi in resp:
                if respi['Errors']:
                    self.throw_worker_error(respi['Message'])
                else:
                    resp_final.append(respi['Value'])
            return resp_final
        else:
            if resp['Errors']:
                self.throw_worker_error(resp['Message'])
            else:
                return resp['Value']


    def rest_call(self, request=None, url=None, **kwargs):
        """
        This is the main entry point for performing any calls to the UQCloud API.
        It is currently just a wrapper of the rest_poll method but in the future 
        it can/should be extended, as polling is just one (out of several) way to implement the 
        submission/status check/updates/results retrieval of a job.

        The request variable should not be confused with any variables directly connected
        to the python requests package. It is instead just a dictionary with the following 
        values (example):
        request = {
                    'Command' : 'uq_subsample',
                    'Argument': [{'Value': X},
                         {'Value': NK},
                         {'Value': Method},
                         {'Value': Name},
                         {'Value': Value}],
                    'nargout' : 2
            }
        where:
            'Command': is a string of the uqlab command that will be executed by the API
            'Argument': is a list of dictionaries of the form {'Value': some_value} that contains
                    the input arguments of the 'Command'
            'nargout': the number of output arguments that we need from the execution of the 'Command'
        """
        if url is None:
            url = self.compute_url
        if request is None:
            raise ValueError('Empty request, aborting rest call.')
        if self.session_type == 'cloud':
            return self.rest_poll(request=request, **kwargs)
        else:
            raise ValueError(f"Unsupported session_type = {self.session_type}.")

    def process_intermid_response(self, resp):
        """
        The goal of this function is to process a response from the API when intermediate computations
        need to be provided. In that case, the response (that maybe a JSON or a MAT file) should be 
        translated to a dictionary that looks like this:
        FROM resp (in .json or .mat format)
        TO resp_dict = {
            'function': '...',
            'data': [...]
        }
        """
        pass

        
    def rest_poll(self, request, **kwargs):
        """
        Polling-based communication with a UQCloud API.
        The logic of polling is simple: 
        1) We send a request for some computation to the URL self.compute_url. The API 
           responds with the jobID of the submitted comutation job. The jobID allows us to
           keep track of the progress of the submitted job. 
        2) Every X seconds (where X=self.refresh_period) we communicate with the API URL 
           self.update_url to get the status of the computation job with ID = jobID. While
           the job is still in progress the API will respond with HTTP code 202. This part
           of periodically calling the API is called polling.
        3.1) While polling, if the job is halted because of an intermediate computation the 
           API responds with HTTP code 201. In that case we have to carry out locally the 
           intermediate computation and then submit the results to the URL self.update_intermid_url.
           Then the job (that we submitted in step 1) will continue and we continue polling.
        3.2) When the job is finished the API responds with code HTTP 200 and within the same 
           response we have the contents of the job results.
        
        Additional notes:    
        - The polling of the job will continue until the self.timeout time has been reached. If
          the job is not complete by then we raise a timeout error. 
        - Every time an intermediate job is needed, the timeout timer is reset. 
        """
        # Determine the request and response formats
        if 'request_format' in kwargs:
            self.logger.debug(f"Found request_format={kwargs['request_format']} in kwargs of rest_poll.")
            request_format_parsed = kwargs['request_format']
        else:
            request_format_parsed = self.default_request_format
            self.logger.debug(f"Using the default request_format = {request_format_parsed}")
        if 'response_format' in kwargs:
            self.logger.debug(f"Found response_format={kwargs['response_format']} in kwargs of rest_poll.")
            response_format_parsed = kwargs['response_format']
        else:
            response_format_parsed = self.default_response_format
            self.logger.debug(f"Using the default response_format = {response_format_parsed}")
        if 'timed' in kwargs:
            self.logger.debug("Will perform timed execution of the request as requested by the user")
            do_timed_execution = kwargs['timed']
        else:
            do_timed_execution = self.default_timed_execution

        # send request
        (scode, resp, resp_dtype) = self.send_request(url=self.compute_url, request=request, 
                                        request_format=request_format_parsed,
                                        response_format=response_format_parsed,
                                        do_timed_execution=do_timed_execution)
        if scode == 201 or scode == 200:
            resp_json = resp
            jobID = int(resp_json['jobID'])
        else:
            jobID = -2

        if jobID < 0:
            self.throw_worker_error('UQ Engine error: the submitted request could not be completed.')

        update_url = f'{self.update_url}{str(jobID)}'
        timeout = time.time() + self.timeout
        progcounter = self.progcounter
        processing_flag = False
        # wait until timeout
        while True:
            (scode, resp, resp_dtype) = self.send_request(url=update_url, request='', request_format='JSON',
                                        response_format=response_format_parsed)
            if scode == 200:
                # return result
                if processing_flag:
                    print(" done!\n")
                return resp
            if scode == 201:
                if processing_flag:
                   print(" done!\n")
                   processing_flag = False
                
                progcounter = self.progcounter

                # there is an intermediate step
                intermid_info_json = resp
                intermid_fun = intermid_info_json['function']
                intermid_input = intermid_info_json['data']
                self.logger.info(f'Received intermediate compute request, function: {intermid_fun}.')
                self.logger.info('Carrying out local computation...')
                if 'parameters' in intermid_info_json:
                    intermid_parameters = intermid_info_json['parameters']
                    res_intermid = helpers.function_eval(intermid_fun, intermid_input, intermid_parameters)
                else:
                    res_intermid = helpers.function_eval(intermid_fun, intermid_input)
                self.logger.info('Local computation complete.')
                req_intermid = {
                    'data': np.array(res_intermid),
                    'jobID': jobID
                }
                self.logger.info('Starting transmission of intermediate compute results ({})...'.
                    format(np.array(res_intermid).shape))
                # Here we need to override any defaults or specified formats for requests/responses. If the remote model
                # expects JSON (resp. MAT) both the requests and responses have to be in JSON (resp. MAT)
                intermid_fmt = resp_dtype.upper()
                if intermid_fmt == 'JSON':
                    req_intermid['data'] = req_intermid['data'].tolist()
                (scode2, resp, resp_dtype) = self.send_request(url=self.update_intermid_url, request=req_intermid, request_format=intermid_fmt,
                                        response_format=intermid_fmt, jobID=jobID, mat_variable_name=MAT_RESPONSE_VARIABLE)
                self.logger.info('Intermediate compute results sent.')
                # reset the timeout timer
                timeout = time.time() + self.timeout
            if time.time() > timeout:
                return {'Errors':True, 'Message':'Timeout reached'}
            time.sleep(self.refresh_period)

            # write a message every self.progcounter polls
            progcounter -= 1
            # print("Current counter: "+str(progcounter))
            if progcounter == 0:
                progcounter = self.progcounter
                if not processing_flag:
                    print("Processing ", end="")
                    processing_flag = True
                print(".", end="")
                

    def send_request(self, url, request, request_format=None, response_format=None, **kwargs):
        """
        This method is responsible for sending a request to a UQCloud API. The request
        may be sent either as a JSON or a binary file. This is determined by the 
        variable communication_mode.
        """
        req_size_kb  = sys.getsizeof(request)/1024
        self.logger.debug(f"Request info | size: {req_size_kb} kb, request_format: {request_format}, response_format: {response_format}")
        if isinstance(request, dict):
            request['OutputFormat'] = response_format.upper()

        if 'compute' in url:
            url_parsed = self.compute_urls_mapper[request_format.lower()]
        elif 'intermid' in url:
            if request_format.lower() == 'mat':
                url_parsed = f"{self.intermid_urls_mapper['mat']}{kwargs['jobID']}/"
            else:
                url_parsed = self.intermid_urls_mapper[request_format.lower()]
        else:
            url_parsed = url
        
        # Handle the case of timed execution
        if 'do_timed_execution' in kwargs and kwargs['do_timed_execution']:
            request['Timed'] = True
            self.logger.debug("Enabling timed execution")

        self.logger.debug(f"Sending request to url: {url_parsed}")
        
        if request_format.lower() == 'json':
            return self.send_json_request(url_parsed, request)
        elif request_format.lower() == 'mat':
            if 'mat_variable_name' in kwargs:
                mat_variable_name_parsed = kwargs['mat_variable_name']
            else:
                mat_variable_name_parsed = None 
            return self.send_mat_request(url_parsed, request, mat_variable_name=mat_variable_name_parsed)
        else:
            raise ValueError(f"Can't handle request_format = {request_format}")
        

    def send_json_request(self, url, request):
        """
        Send a request as JSON
        """
        req_headers = {
            'Authorization': 'Token {}'.format(self.token),
            'Content-Type': 'application/json; charset=utf-8'
        }
        self.logger.debug(request)
        response = requests.post(url, headers=req_headers, data=json.dumps(request, sort_keys=True))
        
        return self.parse_response(response)

    def send_mat_request(self, url, matdata, mat_variable_name=None):
        """
        Send a request as .mat file 
        """
        if mat_variable_name is None:
            mat_variable_name_parsed = MAT_REQUEST_VARIABLE
        else:
            mat_variable_name_parsed = mat_variable_name
        
        matf_bytes = io.BytesIO()

        savemat(matf_bytes, {mat_variable_name_parsed: matdata})
        matf_bytes.seek(0)
        req_headers = {
            'Authorization': 'Token {}'.format(self.token),
        }
        
        response = requests.put(url, headers=req_headers, files={'file': ('matupload.mat', matf_bytes)})
        matf_bytes.close()

        return self.parse_response(response)
    
    def send_empty_get_request(self, url):
        """
        Sends an empty get request to some endpoint and returns the parsed response.
        """
        req_headers = {
            'Authorization': 'Token {}'.format(self.token),
        }
        response = requests.get(url, headers=req_headers)
        return self.parse_response(response)
    
    def parse_response(self, the_response):
        """
        Parses a response retrieved by the UQCloud API.
        The response may be either a binary (mat) file or a string (JSON).
        """
        IS_JSON = False
        IS_MAT = False
        if 'Content-Type' in the_response.headers:
            IS_JSON = the_response.headers['Content-Type'].startswith('application/json')
            IS_MAT = the_response.headers['Content-Type'].startswith('application/octet-stream')
            IS_MAT = IS_MAT or the_response.headers['Content-Type'].startswith('application/x-binary')
        if  IS_JSON:
            resp_body = json.loads(the_response.text)
            self.report_t_elapsed(resp_body)
            return (the_response.status_code, resp_body, 'JSON')
        if IS_MAT:
            mat_bytes = io.BytesIO(the_response.content)
            mat_contents = helpers.loadmat(mat_bytes)
            if MAT_RESPONSE_VARIABLE in mat_contents and isinstance(mat_contents[MAT_RESPONSE_VARIABLE], np.ndarray):
                mat_contents[MAT_RESPONSE_VARIABLE] = helpers._tolist(mat_contents[MAT_RESPONSE_VARIABLE])
    
            # self.logger.debug(f"MAT response contents:{json.dumps(mat_contents, indent=2, default=str)}")
            
            resp_body = mat_contents[MAT_RESPONSE_VARIABLE]
            self.report_t_elapsed(resp_body)
            return (the_response.status_code, resp_body, 'MAT')
        
        raise ValueError(f"Unkown Content-Type: {the_response.headers.get('Content-Type')}")
    def report_t_elapsed(self, response):
        t_elapsed = None 
        if isinstance(response, list):
            if 'tElapsed' in response[0] and response[0]['tElapsed']:
                t_elapsed = response[0]['tElapsed']
        if isinstance(response, dict):
            if 'tElapsed' in response and response['tElapsed']:
                t_elapsed = response['tElapsed']
        if t_elapsed is not None:
            print(f"> Time elapsed in UQLab engine: {t_elapsed} seconds")

class cloud(Singleton, uq_session):
    def __init__(self,host=None, token=None, name=None, 
                 strict_ssl=True,log_level="INFO", log_format=None,
                 force_restart=False):
        super().__init__(name,log_level,log_format)
        if host is None:
            self.host = self.stored_host
        else:
            self.host = host
        if token is None:
            self.token = self.stored_token
        else:
            self.token = token
        self.refresh_period = 1 # in seconds
        self.progcounter = 5 # display a "request in progress" message during polling every self.progcounter polls
        self.timeout = 160 # in seconds
        # the request and response formats can be either 'json' or 'mat'
        self.default_request_format = 'JSON'
        self.default_response_format= 'JSON'
        self.default_timed_execution = False 
        self.compute_url = '{}/compute/'.format(self.host)
        self.compute_mat_url = '{}/compute-mat/'.format(self.host)
        self.update_url = '{}/update/'.format(self.host)
        self.update_intermid_url = '{}/update-intermid/'.format(self.host)
        self.update_intermid_mat_url = '{}/update-intermid-mat/'.format(self.host)
        self.restart_worker_url = '{}/restart-worker/'.format(self.host)
        self.session_type = 'cloud'
        self.compute_urls_mapper = {
            "mat": self.compute_mat_url,
            "json": self.compute_url
        }
        self.intermid_urls_mapper = {
            "mat": self.update_intermid_mat_url,
            "json": self.update_intermid_url
        }
        

        # In case the strict_ssl flag is set to False, requests to the UQCloud
        # API will not involve checking the SSL certificate of the API. This can
        # be used to bypass certificate errors that may randomly occur in some OS's
        if strict_ssl:
            self.ssl_context = None
        else:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            self.ssl_context = ctx
            # Raise a warning to remind the user that this is risky and they should
            # know what they are doing
            warn_msg = ("The SSL certificate of the API host will not be verified." +
            " Make sure that you understand the risks involved!")
            self.logger.warning(warn_msg)

        if force_restart:
            self.hard_reset()

        self.new()

    def new(self):

        self.cli = interface.uq(self)
        REQ = {
            'Command' : 'disp',
            'Argument' : [{'Value':'Session started.'}],
            'nargout' : 0,
        }
        resp = self.rest_call(REQ, request_format='json', response_format='json')
        #print("\nRESP:{}\n".format(resp))
        #print('_____________')
        if resp['Errors']:
            raise RuntimeError(resp['Message'])
        else:
            # self.logger.info('A new session ({}) started.'.format(self.name))
            self.logger.info(f"""This is UQ[py]Lab, version {self.version}, running on {self.host}. 
                                 UQ[py]Lab is free software, published under the open source BSD 3-clause license.
                                 To request special permissions, please contact:
                                  - Stefano Marelli (marelli@ibk.baug.ethz.ch).
                                 A new session ({self.name}) started.""")    
    def rename(self,name):
        self.name = name
    def quit(self):
        REQ = {
            'Command' : 'exit',
            'nargout' : 0,
        }
        #resp = self.rest_call(REQ)
        (scode, resp, resp_dtype) = self.send_request(url=self.compute_url, request=REQ, request_format='json', response_format='json')
        if resp['Errors']:
            self.logger.error('Something went wrong while terminating session {}.'.format(self.name))
            return False
        else:
            self.logger.info('Session {} terminated.'.format(self.name))
            self.name = None
            return True

    def list(self):
        REQ = {
            'Command' : 'uq_list_sessions_web',
        }
        resp = self.rest_call(REQ, request_format='json', response_format='json')
        resp_value = self.get_resp_value(resp)
        print(resp_value)
    def reset(self):
        REQ = {
            'Command' : 'uqlab',
            'nargout' : 0
        }
        resp = self.rest_call(REQ, request_format='json', response_format='json')
        self.logger.info('Reset successful.')
    def hard_reset(self):
        (status_code, body, fmt) = self.send_empty_get_request(self.restart_worker_url)
        SESSION_RESET_HAD_ERRORS = True
        try:
            self.reset()
            SESSION_RESET_HAD_ERRORS = False 
        except Exception as e:
            e_message = str(e)
            pass
        if not body['Errors'] == 200 and not SESSION_RESET_HAD_ERRORS:
            self.logger.info("Succesfully restarted the worker.")
            return
        else:
            if SESSION_RESET_HAD_ERRORS:
                raise RuntimeError(f"Failed to reset the uqlab worker after restart with error message: {e_message}")
            else:
                raise RuntimeError(f"Failed to restart the uqlab worker with error message: {body['Message']}")
    def remove(self, name):
        REQ = {
            'Command' : 'uq_delete_session_web',
            'Argument' : [{'Value':name}],
            'nargout'  : 2
        }
        resp = self.rest_call(REQ, request_format='json', response_format='json')
        resp_value = self.get_resp_value(resp)
        if name == '*':
            self.logger.info('All sessions removed.')
        else:
            self.logger.info('Session {}  removed.'.format(name))

    def save(self, name=None):
        if name is None:
            session_name = self.name
        else:
            session_name = name

        REQ = {
            'Command' : 'uq_save_session_web',
            'Argument' : [{'Value':session_name}]
        }
        resp = self.rest_call(REQ, request_format='json', response_format='json')
        resp_value = self.get_resp_value(resp)
        self.logger.info('Session {} saved.'.format(session_name))
    def load(self, name):
        REQ = {
            'Command' : 'uq_load_session_web',
            'Argument' : [{'Value':name}]
        }
        resp = self.rest_call(REQ, request_format='json', response_format='json')
        resp_value = self.get_resp_value(resp)
        self.name = name
        self.logger.info('Session {} loaded.'.format(name))
    def save_config(self):
        self.logger.info(helpers.uqcloud_save_config(self.host,self.token))
