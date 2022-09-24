import re
from typing_extensions import Required
from metaflow import FlowSpec, step, Flow, Parameter, JSONType

class WineClassifierPredictFlow(FlowSpec):

    vector = Parameter('vector', type=JSONType, help="The vector to predict the class for", Required=True)

    @step
    def start(self):
        run = Flow('WineClassifierTrainFlow').latest_successful_run
        self.train_run_id = run.pathspec
        self.model = run['end'].task.data.model
        print('Input vector: ', self.vector)
        self.next(self.end)

    @step
    def end(self):
        print('Model: ', self.model)

if __name__ == '__main__':
    WineClassifierPredictFlow()