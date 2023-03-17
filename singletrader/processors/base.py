from abc import abstractmethod
from functools import partial


class BaseProcessor():
    pass


class NormalProcessor():

    def __init__(self,**parameters):
        self.params = parameters

    @abstractmethod
    def fit(self,data):
        return
    

    @abstractmethod
    def func(self,data):
        pass

    def __call__(self,data,**parameters):
        return self.fit(data)





class CSProcessor(BaseProcessor):
    def __init__(self):
        pass

    def fit(self,data):
        pass