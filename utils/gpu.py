import tensorflow as tf


def setup_gpu(gpu_ids):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            sel_gpus = [gpus[g] for g in gpu_ids]
            tf.config.experimental.set_visible_devices(sel_gpus, 'GPU')
            for g in sel_gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
