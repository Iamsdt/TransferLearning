import Network as Network
import torch.nn as nn
import torch


def flatten_tensor(x):
    return x.view(x.shape[0], -1)


def load_chkpoint(chkpoint_file):
    restored_data = torch.load(chkpoint_file)

    params = restored_data['params']
    print('load_chkpoint: best accuracy = {:.3f}'.format(params['best_accuracy']))

    if params['model_type'].lower() == 'classifier':
        net = FC(num_inputs=params['num_inputs'],
                 num_outputs=params['num_outputs'],
                 layers=params['layers'],
                 device=params['device'],
                 criterion_name=params['criterion_name'],
                 optimizer_name=params['optimizer_name'],
                 model_name=params['model_name'],
                 lr=params['lr'],
                 dropout_p=params['dropout_p'],
                 best_accuracy=params['best_accuracy'],
                 best_accuracy_file=params['best_accuracy_file'],
                 chkpoint_file=params['chkpoint_file'],
                 class_names=params['class_names']
                 )
        net.load_state_dict(torch.load(params['best_accuracy_file']))
        net.to(params['device'])
        return net


class FC(Network.Network):

    def __init__(self, num_inputs, num_outputs, layers=None, lr=0.003,
                 class_names=None, optimizer_name='Adam', dropout_p=0.2,
                 non_linearity='relu', model_name="model_name",
                 criterion_name='NLLLoss', model_type='classifier',
                 best_accuracy=0., best_accuracy_file='best_accuracy.pth',
                 chkpoint_file='chkpoint_file.pth', device=None):

        super().__init__(device=device)

        if layers is None:
            layers = []

        self.non_linearity = non_linearity
        self.model = nn.Sequential()
        if len(layers) > 0:
            self.model.add_module('fc1', nn.Linear(num_inputs, layers[0]))
            self.model.add_module('relu1', nn.ReLU())
            self.model.add_module('dropout1', nn.Dropout(p=dropout_p, inplace=True))

            for i in range(1, len(layers)):
                self.model.add_module('fc' + str(i + 1), nn.Linear(layers[i - 1], layers[i]))
                self.model.add_module('relu' + str(i + 1), nn.ReLU())
                self.model.add_module('dropout' + str(i + 1), nn.Dropout(p=dropout_p, inplace=True))
                self.model.add_module('out', nn.Linear(layers[-1], num_outputs))
        else:
            self.model.add_module('out', nn.Linear(num_inputs, num_outputs))

        if model_type.lower() == 'classifier' and criterion_name.lower() == 'nllloss':
            self.model.add_module('logsoftmax', nn.LogSoftmax(dim=1))

        # save
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layer_dims = layers
        self.model_type = model_type
        # create dic
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = {str(k): v for k, v in enumerate(list(range(num_outputs)))}

        self.set_model_params(
            criterion_name, optimizer_name, lr, dropout_p, model_name, model_type,
            best_accuracy, best_accuracy_file, chkpoint_file, num_inputs,
            num_outputs, layers, class_names)

    def forward(self, x):
        return self.model(flatten_tensor(x))

    def _get_dropout(self):
        for layer in self.model:
            if type(layer) == torch.nn.modules.dropout.Dropout:
                return layer.p

    def _set_dropout(self, p=0.2):
        for layer in self.model:
            if type(layer) == torch.nn.modules.dropout.Dropout:
                print('FC: setting dropout prob to {:.3f}'.format(p))
                layer.p = p

    def set_model_params(self, criterion_name, optimizer_name, lr,
                         dropout_p, model_name, model_type, best_accuracy,
                         best_accuracy_file, chkpoint_file, num_inputs,
                         num_outputs, layers, class_names):

        super(FC, self).set_model_params(criterion_name, optimizer_name, lr,
                                         dropout_p, model_name,
                                         best_accuracy, best_accuracy_file,
                                         chkpoint_file)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layer_dims = layers
        self.model_type = model_type

        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = {k: str(v) for k, v in enumerate(list(range(num_outputs)))}

    def get_model_params(self):
        params = super(FC, self).get_model_params()
        params['num_inputs'] = self.num_inputs
        params['num_outputs'] = self.num_outputs
        params['layers'] = self.layer_dims
        params['model_type'] = self.model_type
        params['class_names'] = self.class_names
        params['device'] = self.device
        return params
