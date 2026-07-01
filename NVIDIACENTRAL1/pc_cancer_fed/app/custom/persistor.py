import os
import numpy as np
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.abstract.learnable import Learnable

class SimpleNPPersistor(ModelPersistor):
    def __init__(self, save_name="modelo_final_tfm.npy"):
        super().__init__()
        self.save_name = save_name

    def load_model(self, fl_ctx: FLContext):
        
        model_learnable = Learnable()
        model_learnable["weights"] = {} 
        return model_learnable

    def save_model(self, model_learnable, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        run_dir = workspace.get_run_dir(fl_ctx.get_prop(AppConstants.CURRENT_RUN))
        save_path = os.path.join(run_dir, self.save_name)
        
        
        weights = model_learnable.get("weights")
        np.save(save_path, weights)
        self.logger.info(f"--- Model saved in: {save_path} ---")
