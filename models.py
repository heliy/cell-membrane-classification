#coding: UTF-8

n1 = {
    'window_size': 65,
    'input_shape': (1, 65, 65),

    'loss': 'categorical_crossentropy',
    'optimizer': 'adagrad',
    'dropout': 0.1,
    'conve_activa': 'tanh',
    'dense_activa': 'sigmoid',
    'conve_layers': [[48, 4, 4], [36, 5, 5], [36, 4, 4]],
    'pool_sizes': [[2, 2], [2, 2], [2, 2]],
    'dense_layers': [200, 2],    
    }

n4 = {
    'window_size': 95,
    'input_shape': (1, 95, 95),
    
    'loss': 'categorical_crossentropy',
    'optimizer': 'adagrad',
    'dropout': 0.1,
    'conve_activa': 'tanh',
    'dense_activa': 'sigmoid',
    'conve_layers': [[48, 4, 4], [48, 5, 5], [48, 4, 4], [48, 4, 4]],
    'pool_sizes': [[2, 2], [2, 2], [2, 2], [2, 2]],
    'dense_layers': [200, 2],
    }

