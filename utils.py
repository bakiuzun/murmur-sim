import pickle 
from flax import nnx 

def save_model(filepath, params: nnx.statelib.State, non_params:nnx.statelib.State):
    with open(filepath, 'wb') as f:
        pickle.dump({"params": params, "non_params": non_params}, f)

def load_model(filepath, graphdef:nnx.graph.GraphDef):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return nnx.merge(graphdef, data["params"], data["non_params"])
