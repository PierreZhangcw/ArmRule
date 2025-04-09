import logging

logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s %(message)s",
                    datefmt = '%Y-%m-%d %H:%M:%S',
                    )
class Config:
    def __init__(self,gamma=0.07,gamma0=0.1,gamma1=0.8,lam=0.9,data='ficdata2/2',task='',minsup=10):
        #
        self.DATA_PATH = '../'+data+'/'
        self.set_task_path(task)

        #
        self.cores = 25
        self.minsup = minsup
        self.TRAIN_TEST_RATIO = 9
        self.TEST_RATIO = 0.1
        self.TRAIN_RATIO = 0.9
        #
        self.GAMMA = gamma
        self.GAMMA0 = gamma0
        self.GAMMA1 = gamma1
        self.LAMBDA = lam
        #
        self.error_detection = False

    def set_data_path(self,data):

        self.DATA_PATH = '../'+data+'/'
        self.set_task_path()

    def set_task_path(self,task=''):
        if task:
            self.TASK_PATH = self.DATA_PATH+task+'/'
        else:
            self.TASK_PATH = self.DATA_PATH

    def print_config(self):
        dic = {}
        dic['GAMMA'] = self.GAMMA
        dic['GAMMA0'] = self.GAMMA0
        dic['GAMMA1'] = self.GAMMA1
        dic['LAMBDA'] = self.LAMBDA
        return str(dic)
