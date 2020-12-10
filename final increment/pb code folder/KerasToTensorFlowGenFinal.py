
import os

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras.models import load_model

#If a model containing BatchNormization layers
# This line must be executed before loading Keras model.

tf.compat.v1.disable_eager_execution()
K.set_learning_phase(0)
path=os.getcwd()
ModelPath=path+'\inception.h5'
model=keras.models.load_model(ModelPath, compile=False )
# model=keras.models.model_from_json(ModelPath)
model.summary()

print("model.outputs="+str(model.outputs))
print("model.inputs="+str(model.inputs))


from IPython.display import display, HTML,Image
from show_graph import show_graph

import tensorflow as tf
sess = K.get_session()
graph_def = sess.graph.as_graph_def()
# graph_def
show_graph(graph_def)

#################################################################3
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
    



frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.compat.v1.train.write_graph(frozen_graph, "model", path+"/LSTMReg11March_va4TF.pb", as_text=False)
# tf.train.write_graph(frozen_graph, "model", path2+"/nnBased21Feb_vaTF.pb", as_text=False)


