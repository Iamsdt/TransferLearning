import Network as Network
import Network as N
from collections import defaultdict
import torch


class EnsembleModel(Network.Network):

    def __init__(self, models):
        self.criterion = None
        super().__init__()
        self.models = models

    def evaluate(self, testloader, metric='accuracy'):
        # evaluations = defaultdict(float)
        # num_classes = self.models[0][0].num_outputs
        class_correct = defaultdict(int)
        class_totals = defaultdict(int)
        class_names = self.models[0][0].class_names
        with torch.no_grad():
            for inputs, labels in testloader:
                ps_list = []
                for model in self.models:
                    model[0].eval()
                    model[0].to(model[0].device)
                    inputs, labels = inputs.to(model[0].device), labels.to(model[0].device)
                    outputs = model[0].forward(inputs)
                    ps = torch.exp(outputs)
                    ps = ps * model[1]
                    ps_list.append(ps)

                final_ps = ps_list[0]

                for i in range(1, len(ps_list)):
                    final_ps = final_ps + ps_list[i]
                    _, final_preds = torch.max(final_ps, 1)
                    N.update_classwise_accuracies(
                        final_preds, labels, class_correct, class_totals)

        return N.get_accuracies(class_names, class_correct, class_totals)

    def predict(self, inputs, topk=1):
        ps_list = []

        for model in self.models:
            model[0].eval()
            model[0].to(model[0].device)
            with torch.no_grad():
                inputs = inputs.to(model[0].device)
                outputs = model[0].forward(inputs)
                ps_list.append(torch.exp(outputs) * model[1])

        final_ps = ps_list[0]
        for i in range(1, len(ps_list)):
            final_ps = final_ps + ps_list[i]
            _, top = final_ps.topk(topk, dim=1)
            return top

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model[0].forward(x))
        return outputs
