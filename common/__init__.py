import logging, sys, os
import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record) 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(TqdmLoggingHandler())

def set_log(av):
  if av:
    if not os.path.isdir(os.path.dirname(av.logpath)): 
      os.makedirs(os.path.dirname(av.logpath))    
    handler = logging.FileHandler(av.logpath)
    logger.addHandler(handler)
