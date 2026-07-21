"""
Loading of the readily trained detectors from their JSON architecture and HDF5 weight files.

For gradient-based and LRP-based explanations the softmax output has to be removed, so that
relevance is propagated from the raw logit rather than from the normalised probability.
"""
import json


def remove_softmax_from_model(loaded_model_json):
    """Strip the softmax output from a serialised Keras model configuration."""
    dct = json.loads(loaded_model_json)
    layers = dct['config']['layers']

    for i, layer in enumerate(layers):
        if layer['class_name'] == 'Softmax':
            layers.pop(i)
            break

    loaded_model_json = json.dumps(dct)

    # Replace softmax activations declared inline on a layer rather than as their own layer.
    return loaded_model_json.replace('"activation": "softmax"', '"activation": "relu"')


def load_model_and_weights_from_paths(modelpath, weightspath, remove_softmax=False):
    """Build a Keras model from its JSON configuration and load the matching weights."""
    from tensorflow.python.keras.saving.model_config import model_from_json

    with open(modelpath, 'r') as json_file:
        loaded_model_json = json_file.read()

    if remove_softmax:
        loaded_model_json = remove_softmax_from_model(loaded_model_json)

    model = model_from_json(loaded_model_json)
    model.load_weights(weightspath)

    return model


def load_detector(pathology, model_dir, remove_softmax=True):
    """Load the detector for one pathology from the bundled models directory."""
    return load_model_and_weights_from_paths('{}/{}/model.json'.format(model_dir, pathology),
                                             '{}/{}/weights.h5'.format(model_dir, pathology),
                                             remove_softmax=remove_softmax)
