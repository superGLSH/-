import functools
import numpy as np

class Stocks:
    
    def __init__(self, stocks = None, columns = None, returns = None, median = None, risk = None):
        self.stocks = stocks
        self.columns = columns
        self.returns = returns
        self.median = median
        self.risk = risk
        
        self.__height = None
        self.__boundary = None
    
    @staticmethod
    def __import(module_name):
        def decorator(func):
            @functools.wraps(func)
            def _wrapper(*args, **kwargs):
                if not isinstance(module_name, str):
                    raise ValueError("module name can only be string")
                module = __import__(module_name)
                return func(*args, module, **kwargs)
            return _wrapper
        return decorator
    
    @staticmethod
    def __type_checker(attribute_types):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                allArgs = list(args) + list(kwargs.values())
                allArgs[:len(attribute_types)]
                for attr, _type in zip(allArgs, attribute_types):
                    if not isinstance(attr, _type):
                        raise ValueError(f"{attr} should be {_type.__name__} instead of {attr.__class__.__name__}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    @__import('os')
    @__import('pandas')
    @__type_checker([str])
    def __fetch_data(path, key, os, pandas, *args, **kwargs):
        if not isinstance(pandas, __import__('pandas').__class__):
            pandas, args[-1] = args[-1], pandas
            
        def __path_checker(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} does not exist")
            return True
        
        if __path_checker(path):
            
            match key:
                case 'csv':
                    return pandas.read_csv(path, *args, **kwargs)
                case _:
                    raise NotImplemented("unsoported key")
            return 
        return None
    
    @staticmethod
    def __export_data(path, key, pandas = __import__('pandas'),  *args, **kwargs):
        match key:
            case 'csv':
                pandas.to_csv(path, *args, **kwargs)
            case _:
                raise NotImplemented("unsuported key")
        return 
    
    @classmethod
    def fetch_csv(cls, path, *args, **kwargs):
        return cls.__fetch_data(path, 'csv', *args, **kwargs)
    
    @classmethod
    def export_as_csv(cls, path, *args, **kwargs):
        return cls.__export_data(path, 'csv', *args, **kwargs)

    def __initialize_stocks(self, path, *args, **kwargs):
        self.stocks = Stocks.fetch_csv(path)
    
    def __initialize_columns(self):    
        self.columns = np.array(self.stocks.columns.values)
    
    def __initialize_shape(self):
        self.__height = self.stocks.shape[0]
        
    def __initialize_returns(self):
        _transpose = self.stocks.transpose() 
        self.returns = ((np.roll(_transpose, -1) - _transpose) / _transpose)[:, :-1]
    
    def __initialize_median(self):
        self.median = self.returns.sum(1) / (self.__height - 1)

    def __compute_boundary(self):
        self.returns = self.returns[self.median > 0]
        self.columns = self.columns[self.median > 0]
        self.median  = self.median[self.median > 0]
        
    def __compute_risk(self):    
        self.risk = np.array([np.sqrt((self.__height - 1) / (self.__height - 2) * np.sum(np.square(self.returns[i] - self.median[i]))) for i in range(self.median.shape[0])])       
        
    def __to_numpy(self):
        return self.stocks.to_numpy()
    
    def __create_data_frame(self, *args, pandas = __import__('pandas')):
        return pandas.DataFrame(np.array((args)), columns = self.columns)
        

    def __preprocess(self, path, *args, **kwargs):
        
        if self.stocks is None:
            self.__initialize_stocks(path)

        if self.columns is None:
            self.__initialize_columns()
            
        if not isinstance(self.stocks, np.ndarray):
            self.stocks = self.__to_numpy()
            
        if self.__height is None:
            self.__initialize_shape()
            
        if self.returns is None:
            self.__initialize_returns()
            
        if self.median is None:
            self.__initialize_median()
        
        if self.__boundary is None:
            self.__compute_boundary()
        
        if self.risk is None:
            self.__compute_risk()
    
    def preprocess(self, path, name, *args, **kwargs):
        self.__preprocess(path, *args, **kwargs)
        Stocks.export_as_csv(name, self.__create_data_frame(self.median, self.risk), index = None)
