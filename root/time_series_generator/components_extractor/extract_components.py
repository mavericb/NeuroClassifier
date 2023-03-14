from root import constants
from root.time_series_generator.components_extractor.extract_components_DICT import extract_components_DICT
from root.time_series_generator.components_extractor.extract_components_ICA import extract_components_ICA

def extract_components(images_abs_paths, method):
    if method == constants.METHOD_ICA :
        return extract_components_ICA(images_abs_paths)
    else:
        return extract_components_DICT(images_abs_paths)
